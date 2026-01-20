# CLAUDE.md - Instructions for Claude Code

## Project Overview

This is a JavaScript/web port of the Java TAMER (Training an Agent Manually via Evaluative Reinforcement) project. The Java source code is available at `/Users/bradknox/code/tamer_2026/tamerproject/`.

## CRITICAL: Exact Port Requirements

This is an **EXACT PORT** of the Java codebase. When porting or comparing code:

### DO NOT:
- Use WebFetch summaries of Java files - they are often incomplete or summarized
- Guess or infer values for arrays, constants, or parameters
- Assume what values "should be" based on context
- Mark comparisons as complete without verifying exact values
- Rely on tests passing as proof of correctness (tests may not cover all cases)
- **NEVER suggest changing the algorithm, parameters, or behavior from Java** - TAMER training works well in Java for CartPole, MountainCar, Tetris, and LoopMaze. If something isn't working, the bug is in the port, not the algorithm. Do not ask if parameters should be changed.

### ALWAYS:
- Read Java source files directly from `/Users/bradknox/code/tamer_2026/tamerproject/`
- Copy array values, constants, and formulas exactly - character by character
- Compare line-by-line when checking for differences
- If unsure about a value, read the source file again
- Ask the user to visually verify UI/rendering changes (Claude cannot see rendered output)

### Key Java Source Locations:
- Feature generation: `src/edu/utexas/cs/tamerProject/featGen/`
- Models: `src/edu/utexas/cs/tamerProject/modeling/`
- Agents: `src/edu/utexas/cs/tamerProject/agents/`
- Environments: `src/edu/utexas/cs/tamerProject/environments/`
- Credit assignment: `src/edu/utexas/cs/tamerProject/agents/CreditAssign.java`

### When comparing Java to JavaScript:
1. Read the full Java file with the Read tool
2. Extract exact values for all arrays, constants, formulas
3. Compare each value in JavaScript to the Java source
4. Do not paraphrase or summarize - copy exactly
5. Note any intentional differences and get user approval

## JavaScript Structure

- `js/core/` - TamerAgent, HLearner, CreditAssign
- `js/features/` - RBFFeatures
- `js/models/` - LinearModel
- `js/environments/` - LoopMaze, MountainCar, CartPole
- `js/tests/` - Unit tests and learning comparison tests

## Running Tests

```bash
node --experimental-vm-modules js/tests/test.js
node --experimental-vm-modules js/tests/learning_comparison_test.js
```

## Unit Testing Policy

**At every step, consider whether new unit tests are warranted.** When fixing bugs or adding features:

1. Write unit tests that would have caught the bug
2. Add regression tests to prevent the bug from returning
3. Test edge cases around episode boundaries, timing, and credit assignment
4. Verify timing-sensitive behavior (visualization, credit windows, delays)

Key areas requiring test coverage:
- Episode termination and the post-episode pause
- Credit assignment timing (delay, window, extrapolation)
- Visualization timing (when actions/states are shown vs when feedback is credited)
- Reward processing during episode transitions

## Communication Preferences

- When presenting options or alternatives during planning, explain them in normal conversational text - do not use multiple choice format or the AskUserQuestion tool for design discussions
- Reserve AskUserQuestion for gathering specific factual information needed to proceed

## Known Intentional Differences

Document any intentional differences from Java here (must be approved by user):

- (none currently)
