/**
 * LEARNING COMPARISON TESTS
 *
 * These tests verify that JavaScript learning produces IDENTICAL results
 * to Java for the same training sequence.
 *
 * Method: Manually trace through Java formulas step-by-step and verify
 * JavaScript produces the same weights/predictions at each step.
 */

import { RBFFeatures } from '../features/RBFFeatures.js';
import { LinearModel } from '../models/LinearModel.js';

let passCount = 0;
let failCount = 0;

function assertClose(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
        passCount++;
        console.log(`✓ ${message}`);
        return true;
    } else {
        failCount++;
        console.error(`✗ ${message}`);
        console.error(`  Expected: ${expected}, Got: ${actual}, Diff: ${diff}`);
        return false;
    }
}

function assertArrayClose(actual, expected, tolerance, message) {
    let allPass = true;
    for (let i = 0; i < expected.length; i++) {
        if (Math.abs(actual[i] - expected[i]) > tolerance) {
            allPass = false;
            console.error(`  Index ${i}: expected ${expected[i]}, got ${actual[i]}`);
        }
    }
    if (allPass) {
        passCount++;
        console.log(`✓ ${message}`);
    } else {
        failCount++;
        console.error(`✗ ${message}`);
    }
    return allPass;
}

console.log('='.repeat(60));
console.log('LEARNING COMPARISON TESTS');
console.log('Verifying JavaScript produces identical learning to Java');
console.log('='.repeat(60));
console.log('');

// ============================================================
// SETUP: Define exact parameters matching Java defaults
// ============================================================
// From Java FeatGen_RBFs.java:
//   basisFcnsPerDim = 40 (default)
//   relWidth = 0.08
//   normBounds = [-1, 1]
//   biasFeatVal = 0.1
//
// From Java IncGDLinearModel.java:
//   stepSize = 0.001 (default)
//   regL2Wt = 0.0 (default)
//   decayFactor = 0.0 (default, disables eligibility traces)
//
// For testing, use smaller basisFcnsPerDim=3 so we can manually verify

console.log('Test Configuration:');
console.log('  obsRanges = [[0, 10]]');
console.log('  numActions = 2');
console.log('  basisFcnsPerDim = 3');
console.log('  relWidth = 0.08');
console.log('  stepSize = 0.1 (larger for visible learning)');
console.log('');

// ============================================================
// MANUAL JAVA CALCULATION - Step by step
// ============================================================
//
// RBF SETUP:
// normBounds = [-1, 1]
// width = (1 - (-1)) * 0.08 / (3-1) = 2 * 0.08 / 2 = 0.08
//
// For obsRange [0, 10]:
// theObsRangeSize = 10
// dimDistNormFactor = 2 / 10 = 0.2
//
// Means (in raw coordinates):
// normVal = i / (basisFcnsPerDim - 1)
// rawVal = theObsRangeSize * normVal + obsRange[0]
// Mean 0: normVal=0/2=0, rawVal=10*0+0=0
// Mean 1: normVal=1/2=0.5, rawVal=10*0.5+0=5
// Mean 2: normVal=2/2=1, rawVal=10*1+0=10
//
// Features per action = 3 RBFs + 1 bias = 4
// Total features = 4 * 2 actions = 8
//
// FEATURE CALCULATION for obs=[5], action=0:
// Action 0 features are at indices 0-3
//
// For each mean, compute:
// rawDist = obs - mean
// normDist = rawDist * dimDistNormFactor
// sqrdEucDist = normDist^2
// feat = exp(-0.5 * sqrdEucDist / width)
//
// Mean 0 (at 0): rawDist=5, normDist=5*0.2=1, sqrd=1, feat=exp(-0.5*1/0.08)=exp(-6.25)=0.00193045413623...
// Mean 1 (at 5): rawDist=0, normDist=0, sqrd=0, feat=exp(0)=1.0
// Mean 2 (at 10): rawDist=-5, normDist=-1, sqrd=1, feat=exp(-6.25)=0.00193045413623...
// Bias: 0.1
//
// Full feature vector for obs=5, action=0:
// [0.00193045413623, 1.0, 0.00193045413623, 0.1, 0, 0, 0, 0]
//
// Action 1 features would be:
// [0, 0, 0, 0, 0.00193045413623, 1.0, 0.00193045413623, 0.1]

console.log('--- Test 1: Feature Vector Calculation ---');

const rbf = new RBFFeatures([[0, 10]], 2, 3, 0.08);
rbf.setNormBounds(-1, 1);  // TAMER uses normBounds = [-1, 1]
rbf.setBiasFeatPerAct(0.1);  // TAMER uses biasFeatVal = 0.1

// Expected feature values (computed from Java formulas)
const expectedFeat0ForMean0 = Math.exp(-0.5 * 1.0 / 0.08);  // exp(-6.25)
const expectedFeat0ForMean1 = 1.0;  // at center
const expectedFeat0ForMean2 = Math.exp(-0.5 * 1.0 / 0.08);  // exp(-6.25)
const expectedBias = 0.1;

const features_obs5_act0 = rbf.getStateActionFeatures([5], 0);

console.log('Observation: [5], Action: 0');
console.log(`Expected: [${expectedFeat0ForMean0.toFixed(10)}, 1.0, ${expectedFeat0ForMean2.toFixed(10)}, 0.1, 0, 0, 0, 0]`);
console.log(`Got:      [${features_obs5_act0.map(x => x.toFixed(10)).join(', ')}]`);

assertClose(features_obs5_act0[0], expectedFeat0ForMean0, 1e-10, 'Feature[0] = exp(-6.25)');
assertClose(features_obs5_act0[1], 1.0, 1e-10, 'Feature[1] = 1.0 (at mean center)');
assertClose(features_obs5_act0[2], expectedFeat0ForMean2, 1e-10, 'Feature[2] = exp(-6.25)');
assertClose(features_obs5_act0[3], 0.1, 1e-10, 'Feature[3] = 0.1 (bias)');
assertClose(features_obs5_act0[4], 0, 1e-10, 'Feature[4] = 0 (action 1 slot)');
assertClose(features_obs5_act0[5], 0, 1e-10, 'Feature[5] = 0 (action 1 slot)');
assertClose(features_obs5_act0[6], 0, 1e-10, 'Feature[6] = 0 (action 1 slot)');
assertClose(features_obs5_act0[7], 0, 1e-10, 'Feature[7] = 0 (action 1 slot)');

// ============================================================
// Test 2: Single Learning Update
// ============================================================
//
// JAVA gradDescUpdate():
// prediction = sum(weights[i] * feats[i])  // initially 0
// updateEligTraces(feats):
//   for each i: traces[i] *= decayFactor * discountFactor  // 0 with decayFactor=0
//   for each i: traces[i] += feats[i]  // so traces = feats
// err = label - prediction = 1.0 - 0 = 1.0
// wtForErr = stepSize * sampleWeight = 0.1 * 1.0 = 0.1
// for each i:
//   wtedErr = wtForErr * (err - regL2Wt * weights[i]) = 0.1 * (1.0 - 0) = 0.1
//   weights[i] += traces[i] * wtedErr
//
// Weight updates:
// weights[0] += 0.00193045413623 * 0.1 = 0.000193045413623
// weights[1] += 1.0 * 0.1 = 0.1
// weights[2] += 0.00193045413623 * 0.1 = 0.000193045413623
// weights[3] += 0.1 * 0.1 = 0.01
// weights[4..7] += 0 (no change)

console.log('\n--- Test 2: Single Learning Update ---');

const model = new LinearModel(8, {
    stepSize: 0.1,
    regL2Wt: 0.0,
    decayFactor: 0.0
});

// Initial weights should all be 0
const initialWeights = model.getWeights();
assertArrayClose(initialWeights, [0,0,0,0,0,0,0,0], 1e-10, 'Initial weights all zero');

// Initial prediction
const initialPred = model.predict(features_obs5_act0);
assertClose(initialPred, 0.0, 1e-10, 'Initial prediction = 0');

// Perform one update with target=1.0
model.addInstance(features_obs5_act0, 1.0, 1.0);

// Expected weights after update
const f0 = expectedFeat0ForMean0;  // 0.00193045413623
const expectedWeightsAfter1 = [
    f0 * 0.1,     // 0.000193045413623
    1.0 * 0.1,    // 0.1
    f0 * 0.1,     // 0.000193045413623
    0.1 * 0.1,    // 0.01
    0, 0, 0, 0
];

const weightsAfter1 = model.getWeights();
console.log('Weights after 1 update:');
console.log(`Expected: [${expectedWeightsAfter1.map(x => x.toFixed(10)).join(', ')}]`);
console.log(`Got:      [${weightsAfter1.map(x => x.toFixed(10)).join(', ')}]`);

assertArrayClose(weightsAfter1, expectedWeightsAfter1, 1e-10, 'Weights correct after 1 update');

// Prediction after update
// pred = sum(weights * features)
// = f0*0.1*f0 + 1.0*0.1*1.0 + f0*0.1*f0 + 0.1*0.1*0.1 + 0
// = 2*f0^2*0.1 + 0.1 + 0.001
const expectedPredAfter1 = 2 * f0 * f0 * 0.1 + 1.0 * 0.1 + 0.1 * 0.1 * 0.1;
const predAfter1 = model.predict(features_obs5_act0);
console.log(`Prediction after 1 update:`);
console.log(`Expected: ${expectedPredAfter1.toFixed(10)}`);
console.log(`Got:      ${predAfter1.toFixed(10)}`);
assertClose(predAfter1, expectedPredAfter1, 1e-10, 'Prediction correct after 1 update');

// ============================================================
// Test 3: Second Learning Update
// ============================================================
//
// After first update, we have:
// weights = [0.000193045413623, 0.1, 0.000193045413623, 0.01, 0, 0, 0, 0]
// prediction = 2*f0^2*0.1 + 0.1 + 0.001 ≈ 0.101000074549...
//
// Second update:
// err = 1.0 - 0.101000074549 = 0.898999925451
// wtedErr = 0.1 * 0.898999925451 = 0.0898999925451
//
// New weights:
// weights[0] += f0 * 0.0898999925451 = 0.000193045413623 + 0.000173577664...
// weights[1] += 1.0 * 0.0898999925451 = 0.1 + 0.0898999925451 = 0.1898999925451
// etc.

console.log('\n--- Test 3: Second Learning Update ---');

model.addInstance(features_obs5_act0, 1.0, 1.0);

const err2 = 1.0 - expectedPredAfter1;
const wtedErr2 = 0.1 * err2;

const expectedWeightsAfter2 = [
    expectedWeightsAfter1[0] + f0 * wtedErr2,
    expectedWeightsAfter1[1] + 1.0 * wtedErr2,
    expectedWeightsAfter1[2] + f0 * wtedErr2,
    expectedWeightsAfter1[3] + 0.1 * wtedErr2,
    0, 0, 0, 0
];

const weightsAfter2 = model.getWeights();
console.log('Weights after 2 updates:');
console.log(`Expected: [${expectedWeightsAfter2.map(x => x.toFixed(10)).join(', ')}]`);
console.log(`Got:      [${weightsAfter2.map(x => x.toFixed(10)).join(', ')}]`);

assertArrayClose(weightsAfter2, expectedWeightsAfter2, 1e-10, 'Weights correct after 2 updates');

// ============================================================
// Test 4: Learning Different Actions
// ============================================================
console.log('\n--- Test 4: Learning Different Actions ---');

const model2 = new LinearModel(8, {
    stepSize: 0.1,
    regL2Wt: 0.0,
    decayFactor: 0.0
});

// Train action 0 with positive reward
const features_act0 = rbf.getStateActionFeatures([5], 0);
model2.addInstance(features_act0, 1.0, 1.0);

// Train action 1 with negative reward
const features_act1 = rbf.getStateActionFeatures([5], 1);
model2.addInstance(features_act1, -1.0, 1.0);

// Check that action 0 has positive prediction and action 1 has negative
const pred_act0 = model2.predict(features_act0);
const pred_act1 = model2.predict(features_act1);

console.log(`Action 0 prediction: ${pred_act0.toFixed(6)}`);
console.log(`Action 1 prediction: ${pred_act1.toFixed(6)}`);

if (pred_act0 > 0 && pred_act1 < 0) {
    passCount++;
    console.log('✓ Action 0 positive, Action 1 negative after differential training');
} else {
    failCount++;
    console.error('✗ Predictions not differentiated correctly');
}

// ============================================================
// Test 5: Full Training Sequence
// ============================================================
console.log('\n--- Test 5: Full Training Sequence (10 updates) ---');

const model3 = new LinearModel(8, {
    stepSize: 0.1,
    regL2Wt: 0.0,
    decayFactor: 0.0
});

// Manually trace 10 updates
let weights = [0, 0, 0, 0, 0, 0, 0, 0];
const feats = features_obs5_act0;

for (let i = 0; i < 10; i++) {
    // Compute prediction
    let pred = 0;
    for (let j = 0; j < 8; j++) {
        pred += weights[j] * feats[j];
    }

    // Compute error and update
    const err = 1.0 - pred;
    const wtedErr = 0.1 * err;

    for (let j = 0; j < 8; j++) {
        weights[j] += feats[j] * wtedErr;
    }

    // Also update the actual model
    model3.addInstance(feats, 1.0, 1.0);
}

const finalWeights = model3.getWeights();
console.log('Manual computation after 10 updates:');
console.log(`Expected: [${weights.map(x => x.toFixed(10)).join(', ')}]`);
console.log(`Got:      [${finalWeights.map(x => x.toFixed(10)).join(', ')}]`);

assertArrayClose(finalWeights, weights, 1e-9, 'Weights match after 10 updates');

const finalPred = model3.predict(feats);
let expectedFinalPred = 0;
for (let j = 0; j < 8; j++) {
    expectedFinalPred += weights[j] * feats[j];
}
console.log(`Final prediction: expected=${expectedFinalPred.toFixed(6)}, got=${finalPred.toFixed(6)}`);
assertClose(finalPred, expectedFinalPred, 1e-9, 'Final prediction matches');

// ============================================================
// Test 6: Generalization to nearby states
// ============================================================
console.log('\n--- Test 6: Generalization ---');

// Model3 was trained at state 5 with action 0
// Check predictions at nearby states
const pred_state4 = model3.predict(rbf.getStateActionFeatures([4], 0));
const pred_state5 = model3.predict(rbf.getStateActionFeatures([5], 0));
const pred_state6 = model3.predict(rbf.getStateActionFeatures([6], 0));
const pred_state0 = model3.predict(rbf.getStateActionFeatures([0], 0));
const pred_state10 = model3.predict(rbf.getStateActionFeatures([10], 0));

console.log(`Predictions for action 0 at different states:`);
console.log(`  State 0:  ${pred_state0.toFixed(6)}`);
console.log(`  State 4:  ${pred_state4.toFixed(6)}`);
console.log(`  State 5:  ${pred_state5.toFixed(6)} (trained)`);
console.log(`  State 6:  ${pred_state6.toFixed(6)}`);
console.log(`  State 10: ${pred_state10.toFixed(6)}`);

// Trained state should have highest prediction
if (pred_state5 > pred_state4 && pred_state5 > pred_state6) {
    passCount++;
    console.log('✓ Trained state (5) has highest prediction');
} else {
    failCount++;
    console.error('✗ Trained state should have highest prediction');
}

// Nearby states should have higher prediction than far states
if (pred_state4 > pred_state0 && pred_state6 > pred_state10) {
    passCount++;
    console.log('✓ Nearby states generalize (higher than far states)');
} else {
    failCount++;
    console.error('✗ Generalization not working correctly');
}

// ============================================================
// Summary
// ============================================================
console.log('\n' + '='.repeat(60));
console.log(`LEARNING COMPARISON TEST SUMMARY: ${passCount} passed, ${failCount} failed`);
console.log('='.repeat(60));

if (failCount === 0) {
    console.log('\n✓ JavaScript learning matches Java exactly!');
} else {
    console.log('\n✗ MISMATCH DETECTED - JavaScript does not match Java');
}

export { passCount, failCount };
