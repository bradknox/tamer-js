/**
 * Unit Tests for TAMER Web
 * Run in browser by opening test.html or with Node.js
 */

import { RBFFeatures } from '../features/RBFFeatures.js';
import { LinearModel } from '../models/LinearModel.js';

// Simple test framework
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
        console.log(`✓ ${message} (${actual} ≈ ${expected})`);
    } else {
        failCount++;
        console.error(`✗ ${message} (expected ${expected}, got ${actual}, diff=${diff})`);
    }
}

function assertArrayClose(actual, expected, tolerance, message) {
    if (actual.length !== expected.length) {
        failCount++;
        console.error(`✗ ${message} (length mismatch: ${actual.length} vs ${expected.length})`);
        return;
    }
    let allClose = true;
    for (let i = 0; i < actual.length; i++) {
        if (Math.abs(actual[i] - expected[i]) > tolerance) {
            allClose = false;
            console.error(`  Index ${i}: expected ${expected[i]}, got ${actual[i]}`);
        }
    }
    if (allClose) {
        passCount++;
        console.log(`✓ ${message}`);
    } else {
        failCount++;
        console.error(`✗ ${message}`);
    }
}

// ========================================
// RBFFeatures Tests
// ========================================
console.log('\n========== RBFFeatures Tests ==========\n');

// Test 1: Width calculation
// EXACT from Java: width = (normBounds[1] - normBounds[0]) * relWidth / (basisFcnsPerDim - 1)
// With normBounds = [-1, 1]: width = 2 * relWidth / (basisFcnsPerDim - 1)
function testWidthCalculation() {
    console.log('--- Width Calculation ---');

    const obsRanges = [[0, 1]];
    const numActions = 2;

    // Test with default relWidth=0.08, basisFcnsPerDim=5
    // Java TAMER calls setNormBounds(-1, 1), so width = 2 * 0.08 / (5 - 1) = 0.04
    const rbf1 = new RBFFeatures(obsRanges, numActions, 5, 0.08);
    rbf1.setNormBounds(-1, 1);  // Match Java TAMER
    rbf1.setBiasFeatPerAct(0.1);
    const expectedWidth1 = 2 * 0.08 / (5 - 1); // 0.04
    assertClose(rbf1.getWidth(), expectedWidth1, 1e-10, 'Width with basisFcnsPerDim=5, relWidth=0.08');

    // Test with basisFcnsPerDim=40 (Java default for high-resolution)
    const rbf2 = new RBFFeatures(obsRanges, numActions, 40, 0.08);
    rbf2.setNormBounds(-1, 1);  // Match Java TAMER
    rbf2.setBiasFeatPerAct(0.1);
    const expectedWidth2 = 2 * 0.08 / (40 - 1); // ≈ 0.0041
    assertClose(rbf2.getWidth(), expectedWidth2, 1e-10, 'Width with basisFcnsPerDim=40, relWidth=0.08');
}

// Test 2: Number of RBF means
// EXACT from Java: means = basisFcnsPerDim^numObsDims
function testNumMeans() {
    console.log('--- Number of Means ---');

    // 1D observation
    const rbf1D = new RBFFeatures([[0, 1]], 2, 5, 0.08);
    rbf1D.setNormBounds(-1, 1);
    rbf1D.setBiasFeatPerAct(0.1);
    assert(rbf1D.getNumMeans() === 5, 'Number of means for 1D with 5 basis functions');

    // 2D observation
    const rbf2D = new RBFFeatures([[0, 1], [0, 1]], 2, 5, 0.08);
    rbf2D.setNormBounds(-1, 1);
    rbf2D.setBiasFeatPerAct(0.1);
    assert(rbf2D.getNumMeans() === 25, 'Number of means for 2D with 5 basis functions (5x5=25)');

    // 4D observation (like CartPole)
    const rbf4D = new RBFFeatures([[0, 1], [0, 1], [0, 1], [0, 1]], 2, 5, 0.08);
    rbf4D.setNormBounds(-1, 1);
    rbf4D.setBiasFeatPerAct(0.1);
    assert(rbf4D.getNumMeans() === 625, 'Number of means for 4D with 5 basis functions (5^4=625)');
}

// Test 3: Total feature count
// EXACT from Java: numFeatures = (numMeans + (addBiasFeatPerAct ? 1 : 0)) * numActions
function testNumFeatures() {
    console.log('--- Number of Features ---');

    const rbfWithBias = new RBFFeatures([[0, 1]], 4, 5, 0.08);
    rbfWithBias.setNormBounds(-1, 1);
    rbfWithBias.setBiasFeatPerAct(0.1);
    const expectedWithBias = (5 + 1) * 4; // 24
    assert(rbfWithBias.getNumFeatures() === expectedWithBias, 'Num features with bias: (5+1)*4=24');

    const rbfNoBias = new RBFFeatures([[0, 1]], 4, 5, 0.08);
    rbfNoBias.setNormBounds(-1, 1);
    // No setBiasFeatPerAct call - default is no bias
    const expectedNoBias = 5 * 4; // 20
    assert(rbfNoBias.getNumFeatures() === expectedNoBias, 'Num features no bias: 5*4=20');
}

// Test 4: RBF means placement
// EXACT from Java: means are placed at normalized positions [0, 1/(n-1), 2/(n-1), ..., 1]
// then mapped to raw observation range
function testMeansPlacement() {
    console.log('--- Means Placement ---');

    // 1D case with range [0, 10]
    const rbf = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);
    const means = rbf.getMeans();

    // Expected means at: 0, 2.5, 5, 7.5, 10
    const expectedMeans = [[0], [2.5], [5], [7.5], [10]];

    for (let i = 0; i < expectedMeans.length; i++) {
        assertClose(means[i][0], expectedMeans[i][0], 1e-10, `Mean ${i} at correct position`);
    }

    // 2D case with ranges [0, 1] and [-1, 1]
    const rbf2D = new RBFFeatures([[0, 1], [-1, 1]], 2, 3, 0.08);
    rbf2D.setNormBounds(-1, 1);
    rbf2D.setBiasFeatPerAct(0.1);
    const means2D = rbf2D.getMeans();

    // Expected 9 means (3x3 grid)
    assert(means2D.length === 9, '3x3 grid gives 9 means');

    // First mean should be at (0, -1)
    assertClose(means2D[0][0], 0, 1e-10, 'First mean dim 0');
    assertClose(means2D[0][1], -1, 1e-10, 'First mean dim 1');

    // Last mean should be at (1, 1)
    assertClose(means2D[8][0], 1, 1e-10, 'Last mean dim 0');
    assertClose(means2D[8][1], 1, 1e-10, 'Last mean dim 1');
}

// Test 5: Feature activation for observation at mean center
// EXACT from Java: When obs is exactly at a mean center, that RBF should output exp(0) = 1
function testFeatureAtMeanCenter() {
    console.log('--- Feature at Mean Center ---');

    const rbf = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);

    // Observation at mean center (x=5, which is the middle mean)
    const obs = [5];
    const features = rbf.getStateActionFeatures(obs, 0);

    // The mean at index 2 corresponds to x=5, feature[2] should be 1.0
    assertClose(features[2], 1.0, 1e-10, 'RBF at center = 1.0');

    // Bias feature (last feature for action 0, index 5) should be 0.1 (Java default)
    assertClose(features[5], 0.1, 1e-10, 'Bias feature = 0.1 (Java default)');
}

// Test 6: State-action feature structure
// EXACT from Java: Features for action a start at featsPerAction * a
function testStateActionFeatureStructure() {
    console.log('--- State-Action Feature Structure ---');

    const rbf = new RBFFeatures([[0, 1]], 4, 3, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);
    // featsPerAction = 3 + 1 = 4
    // total features = 4 * 4 = 16

    const obs = [0.5];

    // Action 0: features at indices 0-3
    const feat0 = rbf.getStateActionFeatures(obs, 0);
    assert(feat0[0] !== 0 || feat0[1] !== 0 || feat0[2] !== 0, 'Action 0 has non-zero features in slots 0-2');
    assert(feat0[3] === 0.1, 'Action 0 bias at index 3 = 0.1');
    assert(feat0[4] === 0 && feat0[5] === 0, 'Action 0 has zero features in action 1 slots');

    // Action 2: features at indices 8-11
    const feat2 = rbf.getStateActionFeatures(obs, 2);
    assert(feat2[0] === 0 && feat2[1] === 0, 'Action 2 has zero features in action 0 slots');
    assert(feat2[8] !== 0 || feat2[9] !== 0 || feat2[10] !== 0, 'Action 2 has non-zero features in slots 8-10');
    assert(feat2[11] === 0.1, 'Action 2 bias at index 11 = 0.1');
}

// Test 7: Feature values change with state
// This tests that predictions will differ across states
function testFeaturesChangeWithState() {
    console.log('--- Features Change with State ---');

    const rbf = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);

    const feat0 = rbf.getStateActionFeatures([0], 0);
    const feat5 = rbf.getStateActionFeatures([5], 0);
    const feat10 = rbf.getStateActionFeatures([10], 0);

    // Features should be different
    let anyDifferent01 = false;
    let anyDifferent12 = false;
    for (let i = 0; i < 5; i++) {
        if (Math.abs(feat0[i] - feat5[i]) > 0.01) anyDifferent01 = true;
        if (Math.abs(feat5[i] - feat10[i]) > 0.01) anyDifferent12 = true;
    }

    assert(anyDifferent01, 'Features differ between state 0 and state 5');
    assert(anyDifferent12, 'Features differ between state 5 and state 10');

    // At x=0, the first RBF (mean=0) should be highest
    assert(feat0[0] > feat0[2], 'At x=0, first RBF > middle RBF');

    // At x=10, the last RBF (mean=10) should be highest
    assert(feat10[4] > feat10[2], 'At x=10, last RBF > middle RBF');
}

// ========================================
// LinearModel Tests
// ========================================
console.log('\n========== LinearModel Tests ==========\n');

// Test 8: Initial prediction is zero
function testInitialPrediction() {
    console.log('--- Initial Prediction ---');

    const model = new LinearModel(5, { stepSize: 0.01 });
    const features = [1, 0, 1, 0, 1];

    assertClose(model.predict(features), 0.0, 1e-10, 'Initial prediction is zero');
}

// Test 9: Single update increases prediction toward target
// EXACT from Java: weights[i] += traces[i] * stepSize * sampleWeight * (target - prediction)
// With decayFactor=0, traces[i] = feats[i]
function testSingleUpdate() {
    console.log('--- Single Update ---');

    const model = new LinearModel(3, { stepSize: 0.1, decayFactor: 0.0 });
    const features = [1, 0, 0.5];

    // Target = 1.0, initial prediction = 0
    model.addInstance(features, 1.0, 1.0);

    // After update:
    // error = 1.0 - 0.0 = 1.0
    // weights[0] += 1 * 0.1 * 1.0 * 1.0 = 0.1
    // weights[1] += 0 * 0.1 * 1.0 * 1.0 = 0
    // weights[2] += 0.5 * 0.1 * 1.0 * 1.0 = 0.05

    const weights = model.getWeights();
    assertClose(weights[0], 0.1, 1e-10, 'Weight 0 after single update');
    assertClose(weights[1], 0.0, 1e-10, 'Weight 1 after single update (zero feature)');
    assertClose(weights[2], 0.05, 1e-10, 'Weight 2 after single update (0.5 feature)');

    // Prediction should now be: 0.1*1 + 0*0 + 0.05*0.5 = 0.125
    assertClose(model.predict(features), 0.125, 1e-10, 'Prediction after update');
}

// Test 10: Multiple updates converge toward target
function testConvergence() {
    console.log('--- Convergence ---');

    const model = new LinearModel(2, { stepSize: 0.5, decayFactor: 0.0 });
    const features = [1, 1];

    // Train toward target of 1.0
    for (let i = 0; i < 20; i++) {
        model.addInstance(features, 1.0, 1.0);
    }

    // Should converge close to 1.0
    const prediction = model.predict(features);
    assert(prediction > 0.9 && prediction < 1.1, `Converged to ~1.0 (got ${prediction.toFixed(4)})`);
}

// Test 11: Different features learn different targets
function testDifferentFeaturesDifferentTargets() {
    console.log('--- Different Features Different Targets ---');

    const model = new LinearModel(4, { stepSize: 0.3, decayFactor: 0.0 });

    // Feature set 1 (one-hot for action 0): [1, 0, 0, 0] -> target 1.0
    // Feature set 2 (one-hot for action 1): [0, 0, 1, 0] -> target -1.0
    const feat1 = [1, 0, 0, 0];
    const feat2 = [0, 0, 1, 0];

    // Train alternating
    for (let i = 0; i < 30; i++) {
        model.addInstance(feat1, 1.0, 1.0);
        model.addInstance(feat2, -1.0, 1.0);
    }

    const pred1 = model.predict(feat1);
    const pred2 = model.predict(feat2);

    assert(pred1 > 0.5, `Feature set 1 predicts positive (got ${pred1.toFixed(3)})`);
    assert(pred2 < -0.5, `Feature set 2 predicts negative (got ${pred2.toFixed(3)})`);
}

// Test 12: Sample weight affects update magnitude
function testSampleWeight() {
    console.log('--- Sample Weight ---');

    const model1 = new LinearModel(1, { stepSize: 0.1, decayFactor: 0.0 });
    const model2 = new LinearModel(1, { stepSize: 0.1, decayFactor: 0.0 });

    // Same update but different sample weights
    model1.addInstance([1], 1.0, 1.0);
    model2.addInstance([1], 1.0, 2.0);

    const weights1 = model1.getWeights();
    const weights2 = model2.getWeights();

    assertClose(weights2[0], weights1[0] * 2, 1e-10, 'Double sample weight = double weight change');
}

// Test 13: Batch update processes all samples
function testBatchUpdate() {
    console.log('--- Batch Update ---');

    const model = new LinearModel(2, { stepSize: 0.1, decayFactor: 0.0 });

    const samples = [
        { features: [1, 0], target: 1.0, weight: 1.0 },
        { features: [1, 0], target: 1.0, weight: 1.0 },
        { features: [0, 1], target: -1.0, weight: 1.0 }
    ];

    model.batchUpdate(samples);

    assert(model.numUpdates === 3, 'Batch update processed 3 samples');
}

// ========================================
// Integration Tests
// ========================================
console.log('\n========== Integration Tests ==========\n');

// Test 14: Full TAMER learning cycle
function testFullLearningCycle() {
    console.log('--- Full TAMER Learning Cycle ---');

    // LoopMaze-like: 2D observation [x, y], 4 actions
    const obsRanges = [[0, 5], [0, 5]];  // 6x6 grid to match Java LoopMaze
    const numActions = 4;
    const rbf = new RBFFeatures(obsRanges, numActions, 5, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);
    const model = new LinearModel(rbf.getNumFeatures(), { stepSize: 0.1, decayFactor: 0.0 });

    // Simulate human training: reward "right" action (0) when at start
    // Java LoopMaze actions: 0=right, 1=left, 2=down, 3=up
    const startState = [4, 0];
    const rightAction = 0;

    // Give positive reward for going right from start
    const rightFeatures = rbf.getStateActionFeatures(startState, rightAction);
    for (let i = 0; i < 20; i++) {
        model.addInstance(rightFeatures, 1.0, 1.0);
    }

    // Check that right action is preferred at start
    const predictions = [];
    for (let a = 0; a < numActions; a++) {
        const feat = rbf.getStateActionFeatures(startState, a);
        predictions.push(model.predict(feat));
    }

    const rightPred = predictions[rightAction];
    const otherPreds = predictions.filter((_, i) => i !== rightAction);
    const maxOther = Math.max(...otherPreds);

    assert(rightPred > maxOther, `Right action (${rightPred.toFixed(3)}) preferred over others (max=${maxOther.toFixed(3)})`);
}

// Test 15: Generalization - similar states have similar predictions
function testGeneralization() {
    console.log('--- Generalization ---');

    const obsRanges = [[0, 10]];
    const numActions = 2;
    const rbf = new RBFFeatures(obsRanges, numActions, 5, 0.08);
    rbf.setNormBounds(-1, 1);
    rbf.setBiasFeatPerAct(0.1);
    const model = new LinearModel(rbf.getNumFeatures(), { stepSize: 0.2, decayFactor: 0.0 });

    // Train at x=5, action=0 with positive reward
    const trainState = [5];
    const trainFeatures = rbf.getStateActionFeatures(trainState, 0);
    for (let i = 0; i < 20; i++) {
        model.addInstance(trainFeatures, 1.0, 1.0);
    }

    // Check prediction at trained state
    const predAt5 = model.predict(rbf.getStateActionFeatures([5], 0));

    // Check prediction at nearby state (should also be positive due to RBF overlap)
    const predAt4 = model.predict(rbf.getStateActionFeatures([4], 0));
    const predAt6 = model.predict(rbf.getStateActionFeatures([6], 0));

    // Check prediction at far state (should be much lower)
    const predAt0 = model.predict(rbf.getStateActionFeatures([0], 0));
    const predAt10 = model.predict(rbf.getStateActionFeatures([10], 0));

    assert(predAt5 > 0.5, `Trained state has high prediction (${predAt5.toFixed(3)})`);
    assert(predAt4 > predAt0, `Nearby state (4) higher than far state (0)`);
    assert(predAt6 > predAt10, `Nearby state (6) higher than far state (10)`);
}

// Run all tests
function runAllTests() {
    console.log('Starting TAMER Unit Tests\n');
    console.log('='.repeat(50));

    // RBF tests
    testWidthCalculation();
    testNumMeans();
    testNumFeatures();
    testMeansPlacement();
    testFeatureAtMeanCenter();
    testStateActionFeatureStructure();
    testFeaturesChangeWithState();

    // LinearModel tests
    testInitialPrediction();
    testSingleUpdate();
    testConvergence();
    testDifferentFeaturesDifferentTargets();
    testSampleWeight();
    testBatchUpdate();

    // Integration tests
    testFullLearningCycle();
    testGeneralization();

    // Summary
    console.log('\n' + '='.repeat(50));
    console.log(`\nTest Summary: ${passCount} passed, ${failCount} failed`);
    console.log('='.repeat(50));

    return failCount === 0;
}

// Export for use
export { runAllTests };

// Run if loaded directly
runAllTests();
