/**
 * Java Comparison Unit Tests for TAMER Web
 *
 * These tests verify that the JavaScript implementation produces IDENTICAL
 * results to the Java implementation for the same inputs.
 *
 * Expected values are computed from the Java source code formulas.
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
    } else {
        failCount++;
        console.error(`✗ ${message}`);
        console.error(`  Expected: ${expected}, Got: ${actual}, Diff: ${diff}`);
    }
}

console.log('========== Java Comparison Tests ==========\n');
console.log('These tests verify JavaScript matches Java output for identical inputs.\n');

// ============================================================
// TEST 1: Width Calculation (EXACT Java formula)
// ============================================================
// Java: this.width = (normBounds[1] - normBounds[0]) * this.relWidth / (basisFcnsPerDim - 1)
// With normBounds = [-1, 1]: width = 2 * relWidth / (basisFcnsPerDim - 1)
console.log('--- Test 1: Width Calculation (Java formula) ---');

function testWidthJava() {
    // Test case: basisFcnsPerDim=5, relWidth=0.08
    // Java: width = 2 * 0.08 / (5-1) = 0.16 / 4 = 0.04
    const rbf = new RBFFeatures([[0, 10]], 2, 5, 0.08, true);
    const expectedWidth = 2 * 0.08 / (5 - 1);  // 0.04
    assertClose(rbf.getWidth(), expectedWidth, 1e-10,
        `Width = 2 * 0.08 / 4 = ${expectedWidth}`);
}
testWidthJava();

// ============================================================
// TEST 2: Normalization Factor (EXACT Java formula)
// ============================================================
// Java: dimDistNormFactor[i] = (normBounds[1] - normBounds[0]) / theObsRangeSizes[i]
// With normBounds = [-1, 1]: dimDistNormFactor = 2 / obsRangeSize
console.log('\n--- Test 2: Normalization Factor (Java formula) ---');

function testNormFactorJava() {
    // For obsRange [0, 10]: normFactor = 2 / 10 = 0.2
    const rbf = new RBFFeatures([[0, 10]], 2, 3, 0.08, true);
    // Access internal state to verify
    const expectedNormFactor = 2 / 10;  // 0.2
    assertClose(rbf.dimDistNormFactor[0], expectedNormFactor, 1e-10,
        `normFactor for range [0,10] = 2/10 = ${expectedNormFactor}`);

    // For obsRange [-1, 1]: normFactor = 2 / 2 = 1.0
    const rbf2 = new RBFFeatures([[-1, 1]], 2, 3, 0.08, true);
    assertClose(rbf2.dimDistNormFactor[0], 1.0, 1e-10,
        `normFactor for range [-1,1] = 2/2 = 1.0`);
}
testNormFactorJava();

// ============================================================
// TEST 3: RBF Feature Values (EXACT Java formula)
// ============================================================
// Java: feat = Math.exp((-0.5 * sqrdEucDist) / this.width)
// sqrdEucDist = sum((rawDist * normFactor)^2)
console.log('\n--- Test 3: RBF Feature Values (Java formula) ---');

function testRBFValuesJava() {
    // Setup: obsRange=[0,10], 3 RBFs, relWidth=0.08
    // width = 2 * 0.08 / 2 = 0.08
    // normFactor = 2 / 10 = 0.2
    // Means at: 0, 5, 10

    const rbf = new RBFFeatures([[0, 10]], 2, 3, 0.08, true);
    const width = 2 * 0.08 / (3 - 1);  // 0.08
    const normFactor = 2 / 10;  // 0.2

    // Observation at x=5 (middle mean)
    const obs = [5];
    const features = rbf.getStateActionFeatures(obs, 0);

    // RBF 0 (mean at 0):
    // rawDist = 5 - 0 = 5
    // normDist = 5 * 0.2 = 1.0
    // sqrdEucDist = 1.0
    // feat = exp(-0.5 * 1.0 / 0.08) = exp(-6.25)
    const expectedFeat0 = Math.exp(-0.5 * 1.0 / width);
    assertClose(features[0], expectedFeat0, 1e-10,
        `RBF at mean=0 for obs=5: exp(-6.25) = ${expectedFeat0.toFixed(6)}`);

    // RBF 1 (mean at 5):
    // rawDist = 5 - 5 = 0
    // sqrdEucDist = 0
    // feat = exp(0) = 1.0
    assertClose(features[1], 1.0, 1e-10,
        `RBF at mean=5 for obs=5: exp(0) = 1.0`);

    // RBF 2 (mean at 10):
    // rawDist = 5 - 10 = -5
    // normDist = -5 * 0.2 = -1.0
    // sqrdEucDist = 1.0
    // feat = exp(-0.5 * 1.0 / 0.08) = exp(-6.25)
    const expectedFeat2 = Math.exp(-0.5 * 1.0 / width);
    assertClose(features[2], expectedFeat2, 1e-10,
        `RBF at mean=10 for obs=5: exp(-6.25) = ${expectedFeat2.toFixed(6)}`);

    // Bias feature (Java: biasFeatVal = 0.1)
    assertClose(features[3], 0.1, 1e-10,
        `Bias feature = 0.1 (Java default)`);
}
testRBFValuesJava();

// ============================================================
// TEST 4: 2D RBF Features (Java grid generation)
// ============================================================
console.log('\n--- Test 4: 2D RBF Features (Java grid) ---');

function test2DRBFJava() {
    // 2D: obsRanges = [[0, 8], [0, 8]] (like LoopMaze)
    // 3 RBFs per dim -> 9 total means
    const rbf = new RBFFeatures([[0, 8], [0, 8]], 4, 3, 0.08, true);

    // width = 2 * 0.08 / 2 = 0.08
    const width = 2 * 0.08 / 2;
    // normFactor = 2 / 8 = 0.25 for each dim
    const normFactor = 2 / 8;

    // Means grid (in raw coords):
    // Dim 0: 0, 4, 8
    // Dim 1: 0, 4, 8
    // Combined: (0,0), (0,4), (0,8), (4,0), (4,4), (4,8), (8,0), (8,4), (8,8)

    const means = rbf.getMeans();
    assertClose(means[0][0], 0, 1e-10, 'Mean 0 dim 0 = 0');
    assertClose(means[0][1], 0, 1e-10, 'Mean 0 dim 1 = 0');
    assertClose(means[4][0], 4, 1e-10, 'Mean 4 dim 0 = 4 (center)');
    assertClose(means[4][1], 4, 1e-10, 'Mean 4 dim 1 = 4 (center)');
    assertClose(means[8][0], 8, 1e-10, 'Mean 8 dim 0 = 8');
    assertClose(means[8][1], 8, 1e-10, 'Mean 8 dim 1 = 8');

    // Observation at center (4, 4)
    const obs = [4, 4];
    const features = rbf.getStateActionFeatures(obs, 0);

    // Center RBF (index 4) should be 1.0
    assertClose(features[4], 1.0, 1e-10,
        'Center RBF at obs=(4,4) = 1.0');

    // Corner RBF (index 0, mean at 0,0):
    // rawDist dim0 = 4-0 = 4, normDist = 4*0.25 = 1.0
    // rawDist dim1 = 4-0 = 4, normDist = 4*0.25 = 1.0
    // sqrdEucDist = 1.0 + 1.0 = 2.0
    // feat = exp(-0.5 * 2.0 / 0.08) = exp(-12.5)
    const expectedCorner = Math.exp(-0.5 * 2.0 / width);
    assertClose(features[0], expectedCorner, 1e-10,
        `Corner RBF at obs=(4,4): exp(-12.5) = ${expectedCorner.toExponential(4)}`);
}
test2DRBFJava();

// ============================================================
// TEST 5: Linear Model Update (EXACT Java formula)
// ============================================================
// Java IncGDLinearModel.gradDescUpdate():
// err = label - prediction
// wtedErr = stepSize * sampleWeight * (err - regL2Wt * weights[i])
// weights[i] += traces[i] * wtedErr
// With decayFactor=0: traces[i] = feats[i]
console.log('\n--- Test 5: Linear Model Update (Java formula) ---');

function testLinearUpdateJava() {
    // Setup: 3 features, stepSize=0.1, no regularization
    const model = new LinearModel(3, {
        stepSize: 0.1,
        regL2Wt: 0.0,
        decayFactor: 0.0
    });

    const features = [1.0, 0.5, 0.0];
    const target = 1.0;
    const sampleWeight = 1.0;

    // Initial prediction = 0 (all weights = 0)
    // err = 1.0 - 0.0 = 1.0
    // wtedErr = 0.1 * 1.0 * 1.0 = 0.1
    // With decayFactor=0, traces = features
    // weight[0] += 1.0 * 0.1 = 0.1
    // weight[1] += 0.5 * 0.1 = 0.05
    // weight[2] += 0.0 * 0.1 = 0.0

    model.addInstance(features, target, sampleWeight);

    const weights = model.getWeights();
    assertClose(weights[0], 0.1, 1e-10, 'weight[0] = 1.0 * 0.1 = 0.1');
    assertClose(weights[1], 0.05, 1e-10, 'weight[1] = 0.5 * 0.1 = 0.05');
    assertClose(weights[2], 0.0, 1e-10, 'weight[2] = 0.0 * 0.1 = 0.0');

    // Prediction = 0.1*1.0 + 0.05*0.5 + 0*0 = 0.125
    assertClose(model.predict(features), 0.125, 1e-10,
        'prediction = 0.1*1 + 0.05*0.5 = 0.125');
}
testLinearUpdateJava();

// ============================================================
// TEST 6: Multiple Updates Convergence (Java behavior)
// ============================================================
console.log('\n--- Test 6: Convergence Behavior ---');

function testConvergenceJava() {
    const model = new LinearModel(1, {
        stepSize: 0.5,
        regL2Wt: 0.0,
        decayFactor: 0.0
    });

    const features = [1.0];
    const target = 1.0;

    // Each update:
    // err = target - prediction = 1.0 - weights[0]
    // weights[0] += 1.0 * 0.5 * err = 0.5 * (1.0 - weights[0])
    // This is exponential convergence to 1.0

    // After 1 update: w = 0 + 0.5*(1-0) = 0.5
    model.addInstance(features, target, 1.0);
    assertClose(model.getWeights()[0], 0.5, 1e-10, 'After 1 update: w = 0.5');

    // After 2 updates: w = 0.5 + 0.5*(1-0.5) = 0.75
    model.addInstance(features, target, 1.0);
    assertClose(model.getWeights()[0], 0.75, 1e-10, 'After 2 updates: w = 0.75');

    // After 3 updates: w = 0.75 + 0.5*(1-0.75) = 0.875
    model.addInstance(features, target, 1.0);
    assertClose(model.getWeights()[0], 0.875, 1e-10, 'After 3 updates: w = 0.875');
}
testConvergenceJava();

// ============================================================
// TEST 7: Full Pipeline - Same as Java would produce
// ============================================================
console.log('\n--- Test 7: Full Pipeline (RBF + Linear Model) ---');

function testFullPipelineJava() {
    // Simulate exact Java behavior
    const obsRanges = [[0, 10]];
    const numActions = 2;
    const basisFcnsPerDim = 3;
    const relWidth = 0.08;
    const stepSize = 0.1;

    const rbf = new RBFFeatures(obsRanges, numActions, basisFcnsPerDim, relWidth, true);
    const model = new LinearModel(rbf.getNumFeatures(), {
        stepSize: stepSize,
        decayFactor: 0.0
    });

    // Train: positive reward for action 0 at state 5
    const trainObs = [5];
    const trainAction = 0;
    const trainFeatures = rbf.getStateActionFeatures(trainObs, trainAction);

    // Give 10 positive training samples
    for (let i = 0; i < 10; i++) {
        model.addInstance(trainFeatures, 1.0, 1.0);
    }

    // Check predictions
    const predAction0 = model.predict(rbf.getStateActionFeatures([5], 0));
    const predAction1 = model.predict(rbf.getStateActionFeatures([5], 1));

    // Action 0 should have positive prediction, action 1 should be near 0
    console.log(`  Action 0 prediction at state 5: ${predAction0.toFixed(4)}`);
    console.log(`  Action 1 prediction at state 5: ${predAction1.toFixed(4)}`);

    if (predAction0 > 0.5 && predAction1 < 0.01) {
        passCount++;
        console.log('✓ Trained action has high value, untrained action has ~0');
    } else {
        failCount++;
        console.error('✗ Predictions not as expected');
    }

    // Test generalization
    const predNearby = model.predict(rbf.getStateActionFeatures([4], 0));
    const predFar = model.predict(rbf.getStateActionFeatures([0], 0));

    console.log(`  Action 0 prediction at state 4: ${predNearby.toFixed(4)}`);
    console.log(`  Action 0 prediction at state 0: ${predFar.toFixed(6)}`);

    if (predNearby > predFar) {
        passCount++;
        console.log('✓ Nearby state has higher prediction than far state');
    } else {
        failCount++;
        console.error('✗ Generalization not working');
    }
}
testFullPipelineJava();

// ============================================================
// TEST 8: Bias Feature Value (Java default = 0.1)
// ============================================================
console.log('\n--- Test 8: Bias Feature Value ---');

function testBiasValueJava() {
    const rbf = new RBFFeatures([[0, 1]], 2, 3, 0.08, true);
    const features = rbf.getStateActionFeatures([0.5], 0);

    // Bias is at index 3 (after 3 RBF features)
    assertClose(features[3], 0.1, 1e-10, 'Bias feature = 0.1 (Java biasFeatVal)');

    // Action 1 bias is at index 7
    const features1 = rbf.getStateActionFeatures([0.5], 1);
    assertClose(features1[7], 0.1, 1e-10, 'Action 1 bias feature = 0.1');
}
testBiasValueJava();

// Summary
console.log('\n' + '='.repeat(50));
console.log(`\nJava Comparison Test Summary: ${passCount} passed, ${failCount} failed`);
console.log('='.repeat(50));

export { passCount, failCount };
