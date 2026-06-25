// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <glob.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/model/batching_model.h"
#include "cc/model/loader.h"
#include "cc/model/model.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// Mode flag.
DEFINE_string(mode, "play",
              "Mode of operation: 'play', 'append', or 'calculate'.");

// Model flags.
DEFINE_string(models, "",
              "Comma-separated list of model paths or glob patterns "
              "(e.g. '/path/to/*.minigo'). Sorted lexicographically after "
              "expansion. Used in play & append modes.");
DEFINE_string(append_model, "",
              "Single model path to append (append mode only).");
DEFINE_string(anchors, "",
              "Comma-separated list of anchor model paths or glob patterns "
              "(play mode only; expanded paths must be a subset of --models).");
DEFINE_int32(window, 5, "Window width W for pairing (play mode).");

// Game options flags.
DEFINE_bool(resign_enabled, true, "Whether resign is enabled.");
DEFINE_double(resign_threshold, -0.999, "Resign threshold.");
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed.");

// Tree search flags.
DEFINE_int32(num_readouts, 100, "Number of readouts for MCTS.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_double(value_init_penalty, 2.0,
              "New children value initialization penalty.\n"
              "Child value = parent's value - penalty * color, clamped to"
              " [-1, 1].  Penalty should be in [0.0, 2.0].\n"
              "0 is init-to-parent, 2.0 is init-to-loss [default].\n"
              "This behaves similiarly to Leela's FPU \"First Play Urgency\".");

// Inference flags.
DEFINE_string(device, "",
              "Device for inference. For TPUs, pass the gRPC address.");

// Concurrency flags.
DEFINE_int32(parallel_games, 32, "Number of games to play in parallel.");

// I/O flags.
DEFINE_string(positions, "", "Path to starting positions CSV file.");
DEFINE_string(output, "", "Path to the output ledger CSV.");

namespace minigo {
namespace {

// Expand a comma-separated list of paths/globs into a sorted list of paths.
// Entries containing '*' or '?' are expanded via POSIX glob().
// Plain paths are passed through unchanged.
std::vector<std::string> ExpandGlobs(const std::string& flag_value) {
  std::vector<std::string> result;
  std::vector<std::string> tokens = absl::StrSplit(flag_value, ',');
  for (auto& token : tokens) {
    // Trim whitespace.
    while (!token.empty() && token.front() == ' ') token.erase(token.begin());
    while (!token.empty() && token.back() == ' ') token.pop_back();
    if (token.empty()) continue;

    if (token.find('*') != std::string::npos ||
        token.find('?') != std::string::npos) {
      // Glob expansion.
      glob_t glob_result;
      int ret = glob(token.c_str(), GLOB_TILDE | GLOB_NOSORT, nullptr,
                     &glob_result);
      if (ret == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
          result.push_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);
      } else {
        MG_LOG(WARNING) << "Glob pattern matched no files: " << token;
        globfree(&glob_result);
      }
    } else {
      result.push_back(token);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

// A starting position defined by a sequence of moves.
struct StartingPosition {
  std::vector<Coord> moves;
};

// Parse starting positions from CSV file.
// Each line: A1,B2,C3,...  (GTP coordinates, black/white alternate)
// No header.
std::vector<StartingPosition> ParsePositions(const std::string& path) {
  std::string contents;
  MG_CHECK(file::ReadFile(path, &contents))
      << "Failed to read positions file: " << path;

  std::vector<StartingPosition> positions;
  std::istringstream stream(contents);
  std::string line;
  while (std::getline(stream, line)) {
    // Skip empty lines.
    if (line.empty()) continue;
    // Strip trailing \r if present.
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;

    StartingPosition pos;
    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    for (const auto& token : tokens) {
      std::string trimmed(token);
      // Trim whitespace.
      while (!trimmed.empty() && trimmed.front() == ' ') {
        trimmed.erase(trimmed.begin());
      }
      while (!trimmed.empty() && trimmed.back() == ' ') {
        trimmed.pop_back();
      }
      if (trimmed.empty()) continue;
      Coord c = Coord::FromGtp(trimmed);
      MG_CHECK(c != Coord::kInvalid)
          << "Invalid coordinate in positions file: " << trimmed;
      pos.moves.push_back(c);
    }
    positions.push_back(std::move(pos));
  }
  return positions;
}

// Build a Position by replaying an opening sequence.
Position BuildPosition(const StartingPosition& start_pos) {
  Position position(Color::kBlack);
  for (size_t i = 0; i < start_pos.moves.size(); ++i) {
    Color color = (i % 2 == 0) ? Color::kBlack : Color::kWhite;
    position.PlayMove(start_pos.moves[i], color);
  }
  return position;
}

// Result of a single matchup between two models.
struct MatchResult {
  std::string m1_name;
  std::string m2_name;
  int games_played = 0;
  int m1_wins = 0;
  int m2_wins = 0;
};

// Format a MatchResult as a CSV line.
std::string FormatMatchResult(const MatchResult& result) {
  return absl::StrFormat("%s,%s,%d,%d,%d", result.m1_name, result.m2_name,
                         result.games_played, result.m1_wins, result.m2_wins);
}

// Parse a MatchResult from a CSV line.
MatchResult ParseMatchResult(const std::string& line) {
  std::vector<std::string> tokens = absl::StrSplit(line, ',');
  MG_CHECK(tokens.size() == 5) << "Invalid ledger line: " << line;
  MatchResult result;
  result.m1_name = tokens[0];
  result.m2_name = tokens[1];
  result.games_played = std::stoi(tokens[2]);
  result.m1_wins = std::stoi(tokens[3]);
  result.m2_wins = std::stoi(tokens[4]);
  return result;
}

// Play all positions between two models and return a MatchResult.
// m1_path is treated as "M1", m2_path as "M2".
// For each position, plays two games: one where M1 is the next player,
// one where M2 is the next player.
MatchResult PlayPair(const std::string& m1_path, const std::string& m2_path,
                     const std::vector<StartingPosition>& positions) {
  // Set up game options.
  Game::Options game_options;
  game_options.resign_enabled = FLAGS_resign_enabled;
  game_options.resign_threshold = -std::abs(FLAGS_resign_threshold);

  MctsPlayer::Options player_options;
  player_options.virtual_losses = FLAGS_virtual_losses;
  player_options.inject_noise = false;
  player_options.random_seed = FLAGS_seed;
  player_options.tree.value_init_penalty = FLAGS_value_init_penalty;
  player_options.tree.soft_pick_enabled = false;
  player_options.num_readouts = FLAGS_num_readouts;

  // Create the batcher for this pair.
  // We use buffer_count=2 because there are two models.
  BatchingModelFactory batcher(FLAGS_device, 2);

  // Track wins.
  absl::Mutex mu;
  int m1_wins = 0;
  int m2_wins = 0;
  int total_games = 0;

  // Generate all game tasks: for each position, two games.
  // game_idx 2*p+0: M1 plays the "starting" side (next to move)
  // game_idx 2*p+1: M2 plays the "starting" side (next to move)
  int num_game_tasks = positions.size() * 2;
  std::atomic<int> next_task{0};

  // Get model names by loading one instance.
  std::string m1_name;
  std::string m2_name;
  {
    auto model = batcher.NewModel(m1_path);
    m1_name = model->name();
  }
  {
    auto model = batcher.NewModel(m2_path);
    m2_name = model->name();
  }

  MG_LOG(INFO) << "Playing pair: " << m1_name << " vs " << m2_name
               << " (" << num_game_tasks << " games)";

  auto worker = [&](int worker_id) {
    while (true) {
      int task_id = next_task.fetch_add(1);
      if (task_id >= num_game_tasks) break;

      int pos_idx = task_id / 2;
      bool m2_starts = (task_id % 2 == 1);

      const auto& start_pos = positions[pos_idx];
      Position position = BuildPosition(start_pos);

      // Determine who plays the next move based on the starting position.
      // The next player is determined by position.to_play() after replaying
      // the opening moves.
      // "M1 starts" means M1 plays as position.to_play().
      // "M2 starts" means M2 plays as position.to_play().
      //
      // We assign M1 and M2 to black/white based on who "starts" and what
      // color is to_play.
      std::string black_model_path, white_model_path;
      bool m1_is_black;
      if (!m2_starts) {
        // M1 starts = M1 plays as to_play.
        if (position.to_play() == Color::kBlack) {
          m1_is_black = true;
        } else {
          m1_is_black = false;
        }
      } else {
        // M2 starts = M2 plays as to_play.
        if (position.to_play() == Color::kBlack) {
          m1_is_black = false;
        } else {
          m1_is_black = true;
        }
      }

      black_model_path = m1_is_black ? m1_path : m2_path;
      white_model_path = m1_is_black ? m2_path : m1_path;
      std::string black_name = m1_is_black ? m1_name : m2_name;
      std::string white_name = m1_is_black ? m2_name : m1_name;

      Game game(black_name, white_name, game_options);

      auto black_model = batcher.NewModel(black_model_path);
      auto white_model = batcher.NewModel(white_model_path);

      auto black = absl::make_unique<MctsPlayer>(
          std::move(black_model), nullptr, &game, player_options);
      auto white = absl::make_unique<MctsPlayer>(
          std::move(white_model), nullptr, &game, player_options);

      // Initialize to the starting position.
      black->InitializeGame(position);
      white->InitializeGame(position);

      BatchingModelFactory::StartGame(black->model(), white->model());
      auto* curr_player = black.get();
      auto* next_player = white.get();

      while (!game.game_over()) {
        if (curr_player->root()->position.n() >= kMinPassAliveMoves &&
            curr_player->root()->position.CalculateWholeBoardPassAlive()) {
          while (!game.game_over()) {
            MG_CHECK(curr_player->PlayMove(Coord::kPass));
            next_player->PlayOpponentsMove(Coord::kPass);
            std::swap(curr_player, next_player);
          }
          break;
        }

        auto move =
            curr_player->SuggestMove(curr_player->options().num_readouts);
        MG_CHECK(curr_player->PlayMove(move));
        if (move != Coord::kResign) {
          next_player->PlayOpponentsMove(move);
        }
        std::swap(curr_player, next_player);
      }
      BatchingModelFactory::EndGame(black->model(), white->model());

      // Determine winner.
      // game.result() > 0 means black won, < 0 means white won.
      bool m1_won;
      if (game.result() > 0) {
        m1_won = m1_is_black;
      } else {
        m1_won = !m1_is_black;
      }

      {
        absl::MutexLock lock(&mu);
        total_games++;
        if (m1_won) {
          m1_wins++;
        } else {
          m2_wins++;
        }
        if (total_games % 10 == 0 || total_games == num_game_tasks) {
          MG_LOG(INFO) << "  Progress: " << total_games << "/"
                       << num_game_tasks << " games  (" << m1_wins << "-"
                       << m2_wins << ")";
        }
      }
    }
  };

  // Launch workers.
  int num_workers = std::min(FLAGS_parallel_games, num_game_tasks);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_workers; ++i) {
    threads.emplace_back(worker, i);
  }
  for (auto& t : threads) {
    t.join();
  }

  MatchResult result;
  result.m1_name = m1_name;
  result.m2_name = m2_name;
  result.games_played = total_games;
  result.m1_wins = m1_wins;
  result.m2_wins = m2_wins;

  MG_LOG(INFO) << "Result: " << m1_name << " vs " << m2_name << ": "
               << m1_wins << "-" << m2_wins << " (" << total_games
               << " games)";

  return result;
}

// Write a list of match results to a file (overwrite).
void WriteResults(const std::string& path,
                  const std::vector<MatchResult>& results) {
  std::string contents;
  for (const auto& r : results) {
    absl::StrAppend(&contents, FormatMatchResult(r), "\n");
  }
  MG_CHECK(file::WriteFile(path, contents))
      << "Failed to write results to: " << path;
}

// Append a single match result to a file.
void AppendResult(const std::string& path, const MatchResult& result) {
  // Read existing contents, append, write back.
  std::string contents;
  file::ReadFile(path, &contents);  // OK if file doesn't exist yet.
  absl::StrAppend(&contents, FormatMatchResult(result), "\n");
  MG_CHECK(file::WriteFile(path, contents))
      << "Failed to write results to: " << path;
}

// Read match results from a ledger file.
std::vector<MatchResult> ReadResults(const std::string& path) {
  std::string contents;
  MG_CHECK(file::ReadFile(path, &contents))
      << "Failed to read ledger file: " << path;

  std::vector<MatchResult> results;
  std::istringstream stream(contents);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;
    results.push_back(ParseMatchResult(line));
  }
  return results;
}

// Get the basename (stem) of a model path for use as model name.
std::string ModelName(const std::string& path) {
  return std::string(file::Stem(path));
}

// ========================================================================
// Mode: play
// ========================================================================
void RunPlayMode() {
  MG_CHECK(!FLAGS_models.empty()) << "--models must be specified in play mode.";
  MG_CHECK(!FLAGS_positions.empty())
      << "--positions must be specified in play mode.";
  MG_CHECK(!FLAGS_output.empty()) << "--output must be specified in play mode.";

  std::vector<std::string> model_paths = ExpandGlobs(FLAGS_models);
  MG_CHECK(model_paths.size() >= 2)
      << "Need at least 2 models for play mode, got " << model_paths.size();
  MG_LOG(INFO) << "Expanded --models to " << model_paths.size() << " models.";

  std::vector<std::string> anchor_paths;
  if (!FLAGS_anchors.empty()) {
    anchor_paths = ExpandGlobs(FLAGS_anchors);
    MG_LOG(INFO) << "Expanded --anchors to " << anchor_paths.size()
                 << " anchors.";
  }

  auto positions = ParsePositions(FLAGS_positions);
  MG_CHECK(!positions.empty()) << "No starting positions found.";
  MG_LOG(INFO) << "Loaded " << positions.size() << " starting positions.";

  int W = FLAGS_window;
  std::vector<MatchResult> all_results;

  for (size_t i = 0; i < model_paths.size(); ++i) {
    const auto& model = model_paths[i];

    // Collect opponents: window + anchors.
    std::vector<std::string> opponents;
    // Use a set of indices to avoid duplicates.
    std::vector<size_t> opponent_indices;

    // Window opponents: W models strictly before model[i].
    for (size_t j = (i > static_cast<size_t>(W) ? i - W : 0); j < i; ++j) {
      opponent_indices.push_back(j);
    }

    // Anchor opponents.
    if (!anchor_paths.empty()) {
      // Find anchor indices in the model list.
      // We need the oldest anchor strictly before i, and the latest anchor
      // strictly before i.
      int oldest_anchor_idx = -1;
      int latest_anchor_idx = -1;

      for (size_t a = 0; a < model_paths.size() && a < i; ++a) {
        for (const auto& anchor : anchor_paths) {
          if (model_paths[a] == anchor) {
            if (oldest_anchor_idx < 0) {
              oldest_anchor_idx = static_cast<int>(a);
            }
            latest_anchor_idx = static_cast<int>(a);
            break;
          }
        }
      }

      // Add oldest anchor if not already in window.
      if (oldest_anchor_idx >= 0) {
        bool already_in =
            std::find(opponent_indices.begin(), opponent_indices.end(),
                      static_cast<size_t>(oldest_anchor_idx)) !=
            opponent_indices.end();
        if (!already_in) {
          opponent_indices.push_back(
              static_cast<size_t>(oldest_anchor_idx));
        }
      }

      // Add latest anchor if different from oldest and not already in window.
      if (latest_anchor_idx >= 0 && latest_anchor_idx != oldest_anchor_idx) {
        bool already_in =
            std::find(opponent_indices.begin(), opponent_indices.end(),
                      static_cast<size_t>(latest_anchor_idx)) !=
            opponent_indices.end();
        if (!already_in) {
          opponent_indices.push_back(
              static_cast<size_t>(latest_anchor_idx));
        }
      }
    }

    // Play against each opponent.
    for (size_t opp_idx : opponent_indices) {
      auto result = PlayPair(model, model_paths[opp_idx], positions);
      all_results.push_back(result);
    }
  }

  WriteResults(FLAGS_output, all_results);
  MG_LOG(INFO) << "Wrote " << all_results.size() << " match results to "
               << FLAGS_output;
}

// ========================================================================
// Mode: append
// ========================================================================
void RunAppendMode() {
  MG_CHECK(!FLAGS_models.empty())
      << "--models must be specified in append mode.";
  MG_CHECK(!FLAGS_append_model.empty())
      << "--append_model must be specified in append mode.";
  MG_CHECK(!FLAGS_positions.empty())
      << "--positions must be specified in append mode.";
  MG_CHECK(!FLAGS_output.empty())
      << "--output must be specified in append mode.";

  std::vector<std::string> model_paths = ExpandGlobs(FLAGS_models);
  MG_LOG(INFO) << "Expanded --models to " << model_paths.size() << " models.";
  auto positions = ParsePositions(FLAGS_positions);
  MG_CHECK(!positions.empty()) << "No starting positions found.";
  MG_LOG(INFO) << "Loaded " << positions.size() << " starting positions.";

  for (const auto& model : model_paths) {
    auto result = PlayPair(FLAGS_append_model, model, positions);
    AppendResult(FLAGS_output, result);
  }

  MG_LOG(INFO) << "Appended " << model_paths.size() << " match results to "
               << FLAGS_output;
}

// ========================================================================
// Mode: calculate
// ========================================================================
void RunCalculateMode() {
  MG_CHECK(!FLAGS_output.empty())
      << "--output must be specified in calculate mode (ledger location).";

  auto results = ReadResults(FLAGS_output);
  MG_CHECK(!results.empty()) << "No results found in ledger.";

  // Collect all unique model names, preserving first-seen order.
  std::vector<std::string> model_names;
  absl::flat_hash_map<std::string, int> name_to_idx;
  auto get_idx = [&](const std::string& name) -> int {
    auto it = name_to_idx.find(name);
    if (it != name_to_idx.end()) return it->second;
    int idx = model_names.size();
    model_names.push_back(name);
    name_to_idx[name] = idx;
    return idx;
  };

  for (const auto& r : results) {
    get_idx(r.m1_name);
    get_idx(r.m2_name);
  }

  int n = model_names.size();
  MG_LOG(INFO) << "Found " << n << " models in ledger with " << results.size()
               << " matchups.";

  // Build per-model matchup data.
  // For each pair of models, accumulate total games and wins.
  struct Matchup {
    int wins_i = 0;   // wins for model i against model j
    int total = 0;     // total games between i and j
  };
  // matchups[i][j] = how model i did against model j.
  std::vector<std::vector<Matchup>> matchups(n, std::vector<Matchup>(n));

  for (const auto& r : results) {
    int i = name_to_idx[r.m1_name];
    int j = name_to_idx[r.m2_name];
    matchups[i][j].wins_i += r.m1_wins;
    matchups[i][j].total += r.games_played;
    matchups[j][i].wins_i += r.m2_wins;
    matchups[j][i].total += r.games_played;
  }

  // Iterative rating computation.
  const float K = 32.0f;
  std::vector<float> R(n, 0.0f);
  std::vector<float> R_new(n, 0.0f);

  for (int iter = 0; iter < 10000; ++iter) {
    float max_change = 0.0f;
    for (int i = 0; i < n; ++i) {
      float W_total = 0.0f;
      float E_total = 0.0f;
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        if (matchups[i][j].total == 0) continue;
        W_total += matchups[i][j].wins_i;
        float win_rate = static_cast<float>(matchups[i][j].wins_i) /
                         matchups[i][j].total;
        E_total += win_rate;
      }
      R_new[i] = R[i] + K * (W_total - E_total);
      max_change = std::max(max_change, std::abs(R_new[i] - R[i]));
    }
    R = R_new;
    if (max_change < 0.1f) {
      MG_LOG(INFO) << "Converged after " << (iter + 1) << " iterations.";
      break;
    }
  }

  // Offset so R[0] = 0.
  float offset = R[0];
  for (int i = 0; i < n; ++i) {
    R[i] -= offset;
  }

  // Find max name length for alignment.
  size_t max_name_len = 0;
  for (const auto& name : model_names) {
    max_name_len = std::max(max_name_len, name.size());
  }

  // Output ratings.
  std::cout << "\nModel Ratings:\n";
  std::cout << std::string(max_name_len + 12, '-') << "\n";
  for (int i = 0; i < n; ++i) {
    std::cout << absl::StreamFormat("%-*s  %8.1f", max_name_len,
                                    model_names[i], R[i])
              << "\n";
  }
  std::cout << std::string(max_name_len + 12, '-') << "\n";
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed);

  if (FLAGS_mode == "play") {
    minigo::RunPlayMode();
  } else if (FLAGS_mode == "append") {
    minigo::RunAppendMode();
  } else if (FLAGS_mode == "calculate") {
    minigo::RunCalculateMode();
  } else {
    MG_LOG(FATAL) << "Unknown mode: " << FLAGS_mode
                  << ". Must be 'play', 'append', or 'calculate'.";
  }

  return 0;
}
