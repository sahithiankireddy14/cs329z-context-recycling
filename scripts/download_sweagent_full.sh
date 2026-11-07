#!/usr/bin/env bash
set -euo pipefail

source_uri="s3://swe-bench-submissions/verified/20240402_sweagent_gpt4/trajs/"
output_dir="data/raw/sweagent_gpt4_20240402"

mkdir -p "$output_dir/trajs"

aws s3 sync "$source_uri" "$output_dir/trajs" \
  --no-sign-request \
  --only-show-errors

curl -fL --retry 3 --silent --show-error \
  "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/20240402_sweagent_gpt4/results/results.json" \
  -o "$output_dir/results.json"

trajectory_count=$(find "$output_dir/trajs" -type f -name '*.traj' | wc -l | tr -d ' ')
echo "Downloaded ${trajectory_count} trajectories to $output_dir/trajs"
