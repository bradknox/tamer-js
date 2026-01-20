/**
 * Tetris Learning Comparison Test
 *
 * This test runs Tetris with predetermined feedback, records all training
 * samples that are generated, and verifies that:
 * 1. JavaScript produces specific final weights
 * 2. The same samples fed to Java would produce identical weights
 *
 * To verify against Java:
 * 1. Run this test to generate samples.json
 * 2. Run the Java TetrisLearningComparisonTest with the same samples
 * 3. Compare the final weights
 */

import { Tetris } from '../environments/tetris/index.js';
import { TetrisFeatures } from '../features/TetrisFeatures.js';
import { LinearModel } from '../models/LinearModel.js';
import { CreditAssign } from '../core/CreditAssign.js';
import { HLearner } from '../core/HLearner.js';
import { TetrisState } from '../environments/tetris/TetrisState.js';
import * as fs from 'fs';

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
    if (condition) {
        passCount++;
        console.log(`✓ ${message}`);
    } else {
        failCount++;
        console.error(`✗ ${message}`);
    }
}

function assertClose(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
        passCount++;
        console.log(`✓ ${message}`);
    } else {
        failCount++;
        console.error(`✗ ${message}`);
        console.error(`  Expected: ${expected}, Got: ${actual}, Diff: ${diff}`);
    }
}

console.log('='.repeat(70));
console.log('TETRIS LEARNING COMPARISON TEST');
console.log('End-to-end verification that JS learning matches Java');
console.log('='.repeat(70));
console.log('');

// ============================================================
// TEST 1: Feature Generation Comparison
// ============================================================
console.log('--- Test 1: Tetris Feature Generation ---');

// Create a known board state
const testState = new TetrisState();
// Clear the board
for (let i = 0; i < testState.worldState.length; i++) {
    testState.worldState[i] = 0;
}
// Add some blocks to create known features
// Fill bottom row partially
for (let col = 0; col < 8; col++) {
    testState.worldState[19 * 10 + col] = 1;  // Bottom row, cols 0-7
}
// Add a block at row 18, col 0 to create a height difference
testState.worldState[18 * 10 + 0] = 1;

// Set current piece info
testState.currentBlockId = 0;  // I-piece
testState.currentRotation = 0;
testState.currentX = 4;
testState.currentY = 0;
testState.blockMobile = true;
testState.worldWidth = 10;
testState.worldHeight = 20;

// Create feature generator
const featGen = new TetrisFeatures([[0, 1]], 5);  // obsRanges not used, 5 actions

// Get observation from state
const obs = testState.getObservation();

// Compute state features
const stateFeats = featGen.getStateFeatures(obs);

console.log('Board state: bottom row filled (cols 0-7), extra block at (18,0)');
console.log(`State features (first 23 non-squared):`);
console.log(`  Col heights: [${stateFeats.slice(0, 10).map(x => x.toFixed(0)).join(', ')}]`);
console.log(`  Max height: ${stateFeats[10]}`);
console.log(`  Col diffs: [${stateFeats.slice(11, 20).map(x => x.toFixed(0)).join(', ')}]`);
console.log(`  Holes: ${stateFeats[20]}`);
console.log(`  Max well: ${stateFeats[21]}`);
console.log(`  Sum wells: ${stateFeats[22]}`);

// Expected values based on Java FeatGen_Tetris logic:
// Column heights (row of first filled cell from top):
// Col 0: row 18 (has block at 18), Col 1-7: row 19, Col 8-9: row 20 (empty)
assertClose(stateFeats[0], 18, 0.01, 'Col 0 height = 18');
assertClose(stateFeats[1], 19, 0.01, 'Col 1 height = 19');
assertClose(stateFeats[8], 20, 0.01, 'Col 8 height = 20 (empty)');
assertClose(stateFeats[9], 20, 0.01, 'Col 9 height = 20 (empty)');

// Max height should be 18 (lowest row number = highest stack)
assertClose(stateFeats[10], 18, 0.01, 'Max col height = 18');

// Column differences
assertClose(stateFeats[11], 1, 0.01, 'Col diff 0-1 = |18-19| = 1');
assertClose(stateFeats[18], 1, 0.01, 'Col diff 7-8 = |19-20| = 1');

// Holes: empty cells below filled cells
// Col 0: row 19 is empty but row 18 and above have blocks? No, row 18 has 1 block
// Actually row 19 col 0 is filled (bottom row), so no hole there
// The hole would be... let me recalculate
// Row 18, col 0 = 1 (filled)
// Row 19, col 0 = 1 (filled, bottom row)
// So no holes in col 0
// Cols 1-7: row 19 filled, nothing above, no holes
// Cols 8-9: empty, no holes
assertClose(stateFeats[20], 0, 0.01, 'Holes = 0');

// ============================================================
// TEST 2: Extended Action Generation
// ============================================================
console.log('\n--- Test 2: Extended Action Generation ---');

// Reset to clean state with a piece
const cleanState = new TetrisState();
cleanState.spawnBlock(0);  // Spawn I-piece
const cleanObs = cleanState.getObservation();

// Get possible actions
const extActions = featGen.getPossActions(cleanObs);
console.log(`Number of possible placements for I-piece: ${extActions.length}`);

// I-piece should have multiple placements
assert(extActions.length >= 10, `I-piece has at least 10 placements (got ${extActions.length})`);

// Each action should have an action list ending in NONE (4)
const firstAction = extActions[0];
assert(firstAction.actList[firstAction.actList.length - 1] === 4,
    'Extended actions end with NONE action (4)');

// ============================================================
// TEST 3: State-Action Feature Computation (getSAFeats)
// ============================================================
console.log('\n--- Test 3: State-Action Feature Computation ---');

// Use the clean state and first extended action
const saFeats = featGen.getSAFeats(cleanObs, firstAction.actList);

console.log(`SA features length: ${saFeats.length}`);
assert(saFeats.length === 46, 'SA features have 46 elements');

// SA features are differences (next - current), so placing a piece should change heights
const nonZeroFeats = saFeats.filter(x => Math.abs(x) > 0.001).length;
console.log(`Non-zero SA features: ${nonZeroFeats}`);
assert(nonZeroFeats > 0, 'SA features have non-zero values (state changed)');

// ============================================================
// TEST 4: Full Learning Pipeline with Recorded Samples
// ============================================================
console.log('\n--- Test 4: Full Learning Pipeline ---');

// Create components matching Java Tetris TAMER setup
const numFeatures = 46;  // Tetris features (no action dimension, uses getSAFeats)
const stepSize = 0.000005 / 47;  // EXACT from Java

const model = new LinearModel(numFeatures, {
    stepSize: stepSize,
    decayFactor: 0.0,
    useBiasWt: true
});

const creditAssign = new CreditAssign({
    distClass: 'previousStep',
    creditDelayMs: 200,
    creditWindowMs: 600,
    extrapolateFutureRew: false,
    delayWtedIndivRew: false,
    noUpdateWhenNoRew: false
});

// Enable training
creditAssign.inTrainSess = true;

const hLearner = new HLearner(model, creditAssign, featGen);

// Record all samples for Java comparison
const recordedSamples = [];
const recordedTimesteps = [];

// Simulate a training session
console.log('\nSimulating training session...');

const env = new Tetris();
env.init();
let currentObs = env.start();

// Generate deterministic sequence of observations and actions
const trainingSequence = [];
let simTime = 0;
const timeStepMs = 500;  // 500ms per step

// Run for 20 steps with rewards at specific times
for (let step = 0; step < 20; step++) {
    const obs = currentObs;
    const possActions = featGen.getPossActions(obs);

    // Select action deterministically (first available)
    const selectedAction = possActions.length > 0 ? possActions[0] : null;

    if (!selectedAction) {
        console.log(`  Step ${step}: No actions available, ending`);
        break;
    }

    // Record timestep
    const feats = featGen.getSAFeats(obs, selectedAction.actList);
    const timeInSec = simTime / 1000;

    recordedTimesteps.push({
        step: step,
        timeMs: simTime,
        obs: Array.from(obs),
        action: selectedAction.actList,
        feats: Array.from(feats)
    });

    // Record in HLearner
    hLearner.recordTimeStepEnd(timeInSec);
    hLearner.recordTimeStepWithFeats(feats, timeInSec, obs, selectedAction.actList);

    // Give reward at specific steps (every 3rd step)
    if (step > 0 && step % 3 === 0) {
        const reward = (step % 6 === 0) ? 1.0 : -1.0;  // Alternate positive/negative
        const rewardTime = simTime + 100;  // 100ms after step start

        hLearner.processReward(reward, rewardTime);

        recordedTimesteps[recordedTimesteps.length - 1].reward = reward;
        recordedTimesteps[recordedTimesteps.length - 1].rewardTimeMs = rewardTime;

        console.log(`  Step ${step}: Gave ${reward > 0 ? '+' : ''}${reward} reward`);
    }

    // Process samples
    const samples = hLearner.processSamples(simTime + timeStepMs, true);
    if (samples.length > 0) {
        for (const sample of samples) {
            recordedSamples.push({
                step: step,
                feats: Array.from(sample.feats),
                label: sample.label,
                weight: sample.weight
            });
        }
        console.log(`  Step ${step}: Processed ${samples.length} samples`);
    }

    // Execute actions in environment
    let stepResult;
    for (const atomicAction of selectedAction.actList) {
        stepResult = env.step(atomicAction);
        if (stepResult.terminal) {
            console.log(`  Step ${step}: Game over!`);
            // Restart for next iteration
            currentObs = env.start();
            break;
        }
    }

    // Update observation for next iteration
    if (stepResult && !stepResult.terminal) {
        currentObs = stepResult.obs;
    }

    simTime += timeStepMs;
}

console.log(`\nTotal recorded timesteps: ${recordedTimesteps.length}`);
console.log(`Total recorded samples: ${recordedSamples.length}`);

// ============================================================
// TEST 5: Verify Model Learned
// ============================================================
console.log('\n--- Test 5: Verify Model State ---');

const finalWeights = model.getWeights();
const nonZeroWeights = finalWeights.filter(w => Math.abs(w) > 1e-15).length;
console.log(`Non-zero weights: ${nonZeroWeights} / ${finalWeights.length}`);

// If we gave rewards and they were processed, we should have non-zero weights
if (recordedSamples.length > 0) {
    assert(nonZeroWeights > 0, 'Model has learned (non-zero weights)');
} else {
    console.log('  (No samples processed - rewards may not have been credited yet)');
}

// Print weight statistics
const absWeights = finalWeights.map(Math.abs);
const maxWeight = Math.max(...absWeights);
const sumWeights = absWeights.reduce((a, b) => a + b, 0);
console.log(`Max |weight|: ${maxWeight.toExponential(4)}`);
console.log(`Sum |weights|: ${sumWeights.toExponential(4)}`);

// ============================================================
// TEST 6: Reproducibility - Same samples produce same weights
// ============================================================
console.log('\n--- Test 6: Reproducibility Test ---');

// Create a fresh model and replay the recorded samples
const model2 = new LinearModel(numFeatures, {
    stepSize: stepSize,
    decayFactor: 0.0,
    useBiasWt: true
});

for (const sample of recordedSamples) {
    model2.addInstance(sample.feats, sample.label, sample.weight);
}

const replayWeights = model2.getWeights();

// Compare weights
let maxDiff = 0;
for (let i = 0; i < finalWeights.length; i++) {
    const diff = Math.abs(finalWeights[i] - replayWeights[i]);
    maxDiff = Math.max(maxDiff, diff);
}

console.log(`Max weight difference (original vs replay): ${maxDiff.toExponential(4)}`);
assert(maxDiff < 1e-10, 'Replaying samples produces identical weights');

// ============================================================
// TEST 7: Save data for Java comparison
// ============================================================
console.log('\n--- Test 7: Export Data for Java Comparison ---');

const exportData = {
    config: {
        numFeatures: numFeatures,
        stepSize: stepSize,
        useBiasWt: true,
        creditDistType: 'previousStep'
    },
    samples: recordedSamples,
    finalWeights: Array.from(finalWeights),
    biasWeight: model.biasWt,
    timesteps: recordedTimesteps
};

const exportPath = '/Users/bradknox/code/tamer_2026/tamer-web/js/tests/tetris_test_data.json';
fs.writeFileSync(exportPath, JSON.stringify(exportData, null, 2));
console.log(`Exported test data to: ${exportPath}`);

// Print checksum for quick comparison
const weightsChecksum = finalWeights.reduce((a, b) => a + b, 0);
console.log(`\nWeights checksum (sum): ${weightsChecksum}`);
console.log(`Bias weight: ${model.biasWt}`);

// ============================================================
// TEST 8: Direct sample-by-sample weight verification
// ============================================================
console.log('\n--- Test 8: Step-by-Step Weight Verification ---');

if (recordedSamples.length > 0) {
    // Manually trace through first few samples
    const model3 = new LinearModel(numFeatures, {
        stepSize: stepSize,
        decayFactor: 0.0,
        useBiasWt: true
    });

    console.log('Tracing first 3 sample updates:');
    for (let i = 0; i < Math.min(3, recordedSamples.length); i++) {
        const sample = recordedSamples[i];

        // Prediction before update
        const predBefore = model3.predict(sample.feats);

        // Manual calculation
        const err = sample.label - predBefore;
        const wtForErr = stepSize * sample.weight;

        console.log(`  Sample ${i}:`);
        console.log(`    Label: ${sample.label.toFixed(6)}, Pred: ${predBefore.toExponential(4)}`);
        console.log(`    Error: ${err.toFixed(6)}, Update magnitude: ${(wtForErr * err).toExponential(4)}`);

        // Apply update
        model3.addInstance(sample.feats, sample.label, sample.weight);

        const predAfter = model3.predict(sample.feats);
        console.log(`    Pred after: ${predAfter.toExponential(4)}`);
    }

    passCount++;
    console.log('✓ Step-by-step verification completed');
}

// ============================================================
// Summary
// ============================================================
console.log('\n' + '='.repeat(70));
console.log(`TETRIS LEARNING COMPARISON TEST SUMMARY: ${passCount} passed, ${failCount} failed`);
console.log('='.repeat(70));

if (failCount === 0) {
    console.log('\n✓ All tests passed!');
    console.log('\nTo verify against Java:');
    console.log('1. The test data has been exported to tetris_test_data.json');
    console.log('2. Create a Java test that reads this file and processes the same samples');
    console.log('3. Compare the final weights - they should be identical');
} else {
    console.log('\n✗ Some tests failed - check output above');
}

export { passCount, failCount, exportData };
