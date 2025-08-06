# Pgeon Refactoring Plan

This document outlines the plan to refactor the `pgeon` codebase to improve modularity, type safety, and test coverage.

## 1. Project Setup

- [x] Transition to `uv` for dependency management.
  - [x] Update `pyproject.toml` for `uv` compatibility.
  - [x] Generate `uv.lock` file.
- [x] Ensure all existing tests pass before refactoring.

## 2. Code Refactoring and Consolidation

- [x] Deprecate `PolicyGraph` (`src/pgeon/policy_graph.py`).
  - [x] Merge functionality from `PolicyGraph` into `PolicyApproximatorFromBasicObservation`.
  - [x] Refactor `PolicyGraph` users to use `PolicyApproximatorFromBasicObservation`.
  - [x] Remove `src/pgeon/policy_graph.py`.
- [ ] Deprecate `IntentionIntrospector` (`src/pgeon/intention_introspector.py`).
  - [ ] Move intention finding logic into `IntentionAwarePolicyGraph` or a new dedicated class that uses the new abstractions.
  - [ ] Remove `src/pgeon/intention_introspector.py`.
- [ ] Refactor `PolicyApproximator` and `PolicyRepresentation`.
  - [ ] Ensure `PolicyApproximatorFromBasicObservation` correctly and consistently uses a `PolicyRepresentation` instance.
  - [ ] Remove redundant methods between the old `PolicyGraph` and the new approximators.
- [ ] Refactor `ipg_xai.py` for better integration.
  - [ ] Update `IPG_XAI_analyser` to use `PolicyApproximator` and `PolicyRepresentation` instead of `PolicyGraph`.
  - [ ] Ensure type consistency with the new abstractions.

## 3. Testing

- [ ] Update tests to reflect refactoring.
  - [ ] Remove or update tests for deprecated files (`test_policy_graph.py`, `test_intention_introspector.py`).
  - [ ] Enhance tests for `test_policy_approximator.py` and `test_policy_representation.py`.
- [ ] Add new tests to improve coverage.
  - [ ] Write unit tests for new or modified functionality.
  - [ ] Implement end-to-end tests using the `cartpole` environment.

## 4. Finalization

- [ ] Code cleanup and final review.
- [ ] Merge the `refactor/codebase-cleanup` branch into `main`.
