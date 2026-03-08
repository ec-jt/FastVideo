# SPDX-License-Identifier: Apache-2.0
"""Helix parallelism test suite.

This module contains tests for validating Helix parallelism (TP+CP) implementation.

Test Models:
- Wan 1.3B: Baseline (works with both SP and Helix)
- Wan 5B: Medium (tight fit with SP, comfortable with Helix)
- Wan 14B: Large (doesn't fit with SP, requires Helix)

Test Strategy:
1. Run baseline tests with current SP implementation
2. Implement Helix parallelism
3. Run Helix tests and compare SSIM with baseline
"""
