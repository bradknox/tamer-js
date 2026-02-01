# TAMER Web Performance Analysis

## Executive Summary

The maximum step frequency is limited by **three main factors**:

1. **Computational cost per step** (varies by environment)
   - Mountain Car: ~1-3ms per step (1600 RBF features × 4 computations)
   - Tetris: ~5-20ms when selecting new extended action (tree search + 34 predictions)

2. **Browser's setTimeout minimum delay** (~4ms, limits async loop to ~250 iterations/sec)
   - Even `setTimeout(fn, 0)` has a minimum ~4ms delay in browsers
   - This is the spec-mandated minimum for nested setTimeout calls

3. **Rendering overhead** (especially Tetris thumbnails)
   - Tetris creates/destroys ~34 DOM elements every time placements are regenerated

## Detailed Analysis

### 1. Mountain Car Step Breakdown

Each step in Mountain Car involves:

```
_doStep(time)
  └── agent.step(time)
      ├── recordTimeStepEnd(time)             ~0.01ms
      ├── processSamples()                    ~0.01ms
      ├── recordTimeStepStart()
      │   └── getStateActionFeatures(obs, action)  ← 1600 RBF + 1600 Math.exp()
      ├── env.step(action)                    ~0.01ms
      └── selectAction(obs)
          └── predictAllActions(obs, 3)
              └── For each of 3 actions:
                  ├── getStateActionFeatures()  ← 1600 RBF + 1600 Math.exp()
                  └── model.predict()           ← dot product of 4803 features
```

**Key bottlenecks:**
- **RBF Features**: With `basisFcnsPerDim=40` and 2 observation dimensions:
  - 40 × 40 = 1600 RBF basis functions
  - Each requires: distance calculation + `Math.exp()` call
  - Total features: 1600 × 3 actions + 3 bias = 4803 features

- **Feature computation happens 4 times per step:**
  1. Once in `recordTimeStepStart()`
  2. Three times in `predictAllActions()` (once per action)

- **Estimated time per step**: 1-3ms

### 2. Tetris Step Breakdown

Tetris uses `ExtActionAgentWrap` which batches atomic actions into extended actions (piece placements).

```
_doStep(time)
  └── ExtActionAgentWrap.step(time)
      ├── Execute atomic action from currExtendedAction
      ├── env.step(atomicAction)              ~0.05ms
      │
      └── [IF extended action complete, get new one:]
          └── coreAgent.step(time)
              └── _selectExtendedAction(obs)
                  ├── getPossActions(obs)      ← Tree search: O(placements × depth)
                  │   └── Returns ~34 placements
                  │
                  └── predictExtendedActions()
                      └── For each of ~34 placements:
                          ├── getSAFeats()     ← Simulate moves + compute features
                          └── model.predict()  ← dot product of 46 features
```

**Key bottlenecks:**
- **getPossActions() Tree Search**:
  - Explores all possible moves until pieces land
  - Typically finds 30-40 valid placements
  - Each exploration involves: state copy, action simulation, duplicate checking

- **getSAFeats() for each placement**:
  - Applies all atomic actions in sequence
  - Writes block to board
  - Checks for row clears
  - Computes 46 features twice (current and next state)
  - Calculates difference

- **When extended action completes** (~every 10-20 atomic steps):
  - Must compute features for ALL ~34 placements
  - Estimated time: 5-20ms

### 3. Loop Timing Mechanism

The `_runLoopFast()` function uses `setTimeout(fn, 0)`:

```javascript
_runLoopFast() {
    // Calculate steps per frame
    const stepsPerFrame = Math.floor(16 / stepDurationMs);

    // Run multiple steps
    for (let i = 0; i < stepsPerFrame; i++) {
        this._doStep(performance.now());
    }

    // Schedule next batch
    setTimeout(() => this._runLoopFast(), 0);
}
```

**Problem**: `setTimeout(fn, 0)` has a minimum delay of ~4ms in browsers (clamped by spec).

**Impact**:
- Even with 0ms requested, actual interval is 4-5ms
- Maximum async loop frequency: ~200-250 Hz
- If `stepsPerFrame=1`, max rate is ~250 steps/sec regardless of computation time

### 4. Rendering Overhead

Every frame calls `_render()`:
- **Non-Tetris**: Simple canvas drawing (~0.1ms)
- **Tetris with thumbnails**:
  - Creates 34+ DOM elements
  - Creates 34+ canvas elements
  - Draws board state 34+ times
  - Much more expensive (~10-50ms when regenerating)

## Identified Issues

### Issue 1: setTimeout Minimum Delay
**Severity**: High
**Impact**: Limits maximum step frequency to ~250 steps/sec when `stepsPerFrame=1`

### Issue 2: Redundant Feature Computation (Mountain Car)
**Severity**: Medium
**Impact**: Features computed 4x per step instead of necessary 2x

Looking at the code flow:
1. `recordTimeStepStart()` computes features for the action just taken
2. `selectAction()` computes features for all 3 actions to find best

The feature computation in `recordTimeStepStart()` could potentially be cached.

### Issue 3: Tetris getPossActions() Called Every Extended Action
**Severity**: High for Tetris
**Impact**: Tree search runs ~every 10-20 steps, taking 5-20ms

### Issue 4: Tetris getActionValues() Recomputes Everything
**Severity**: Medium for Tetris
**Impact**: UI calls `getActionValues()` which recomputes predictions for all placements

### Issue 5: Tetris Thumbnail Rendering
**Severity**: Medium for Tetris
**Impact**: Regenerates 34+ DOM elements when piece changes

## Potential Optimizations (for future consideration)

1. **For setTimeout limit**: Use Web Workers for computation, or batch more steps

2. **For redundant features**: Cache features computed in `selectAction()` for use in `recordTimeStepStart()`

3. **For Tetris**:
   - Cache placements between atomic steps (already partially done)
   - Avoid regenerating all thumbnails every frame
   - Consider using a single canvas with manual positioning

4. **For RBF**:
   - Pre-compute and cache distance calculations when possible
   - Use TypedArrays consistently for feature vectors

## Measured Performance (Estimated)

Based on code analysis:

| Environment | Step Time | Max Freq | Main Bottleneck |
|-------------|-----------|----------|-----------------|
| Loop Maze   | ~0.2ms    | ~5000/s  | setTimeout minimum |
| Cart Pole   | ~0.5ms    | ~2000/s  | 64 RBF features |
| Mountain Car| ~2ms      | ~500/s   | 1600 RBF features |
| Acrobot     | ~0.5ms    | ~2000/s  | 64 RBF features |
| Robot Arm   | ~0.5ms    | ~2000/s  | 64 RBF features |
| Tetris      | ~10ms*    | ~100/s*  | Tree search + predictions |

*Tetris: Step time varies. Atomic steps are fast (~0.1ms), but selecting a new extended action (every ~15 steps) takes 5-20ms.

## Root Causes

### 1. The `basisFcnsPerDim=40` Setting for Mountain Car

Mountain Car uses 40 basis functions per dimension:
- 2 dimensions × 40 per dim = 40² = **1600 RBF means**
- Each prediction: 1600 distance calculations + 1600 `Math.exp()` calls
- Total features: 1600 × 3 actions + 3 bias = **4803 features**

Compare to Cart Pole which uses `basisFcnsPerDim=8`:
- 4 dimensions × 8 per dim = 8⁴ = 4096 means... wait, that's more!

Actually, let me recalculate. The RBF means are computed recursively:
- Mountain Car (2D, 40/dim): 40² = 1600 means
- Cart Pole (4D, 8/dim): 8⁴ = 4096 means
- Loop Maze (2D, 6/dim): 6² = 36 means

So **Cart Pole actually has MORE RBF means** (4096 vs 1600)!

This suggests the step time difference isn't just RBF count - it may be:
1. Different `relWidth` affecting exp() calculation
2. Different overhead in environment physics
3. Or something else

### 2. The `setTimeout(fn, 0)` Minimum Delay

JavaScript's `setTimeout(fn, 0)` has a minimum delay of ~4ms in browsers. This is specified in the HTML spec for nested timeouts.

The current `_runLoopFast()` tries to work around this by computing multiple steps per setTimeout call:
```javascript
const stepsPerFrame = Math.floor(16 / stepDurationMs);
```

For `stepDurationMs=0.001`, this would be `stepsPerFrame=16000`.
But if each step takes 2ms, running 16000 steps would take 32 seconds!

**The calculation is wrong** - it should be based on actual step time, not target time:
```javascript
// Current (wrong): assumes steps are instant
const stepsPerFrame = Math.floor(16 / stepDurationMs);

// Should be: based on time budget
const timeBudgetMs = 16;  // Target ~60fps
const stepsPerFrame = Math.floor(timeBudgetMs / actualStepTimeMs);
```

### 3. Feature Computation Redundancy

Each step computes features multiple times:

```
selectAction()
  └── predictAllActions() - computes features for ALL actions
      └── getStateActionFeatures() x numActions

recordTimeStepStart()
  └── getStateActionFeatures() - computes features AGAIN for selected action
```

The features computed in `selectAction()` could be cached and reused.

### 4. Tetris Extended Action Selection

When Tetris needs a new piece placement:
1. `getPossActions()` does tree search: ~2-5ms
2. `predictExtendedActions()` for ~34 placements: ~3-10ms
   - Each placement: `getSAFeats()` (~0.1-0.2ms) + `predict()` (~0.01ms)

Total: ~5-15ms every 10-20 atomic steps

## Optimization Priority

1. **HIGH**: Fix `stepsPerFrame` calculation to be based on actual step time
2. **HIGH**: Cache features between `selectAction()` and `recordTimeStepStart()`
3. **MEDIUM**: For Tetris, cache placements/predictions during atomic steps
4. **MEDIUM**: Consider using Web Workers for computation
5. **LOW**: Use TypedArrays consistently for feature vectors
6. **LOW**: Optimize Tetris thumbnail rendering (reuse DOM elements)

## Test Commands

Open `tmp/performance_diagnostic.html` in a browser and run:
- "Run Mountain Car Diagnostic" - detailed timing breakdown
- "Run Tetris Diagnostic" - detailed timing breakdown
- "Run Full Step Diagnostic" - compares all environments

Or in browser console:
```javascript
testTamerApp("mountaincar")
testTamerApp("tetris")
```

Open `tmp/timing_diagnostic.html` for lower-level timing tests:
- "Test setTimeout(0) frequency" - measures actual setTimeout interval
- "Test Pure Computation" - measures step speed without async overhead
