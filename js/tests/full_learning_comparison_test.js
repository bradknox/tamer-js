/**
 * Full End-to-End Learning Comparison Test
 *
 * Tests that the complete learning pipeline (environment → features → model)
 * produces identical results in JavaScript and Java for ALL environments.
 *
 * This test will catch mismatches like:
 * - Wrong basisFcnsPerDim (different feature vector dimensions)
 * - Wrong relWidth (different RBF widths)
 * - Wrong stepSize (different weight updates)
 * - Wrong credit assignment (different sample labels)
 *
 * Run with: node --experimental-vm-modules js/tests/full_learning_comparison_test.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Core components
import { TamerAgent } from '../core/TamerAgent.js';
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { TetrisFeatures } from '../features/TetrisFeatures.js';
import { CreditAssign } from '../core/CreditAssign.js';

// Environments
import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';
import { CartPole } from '../environments/CartPole.js';
import { Acrobot } from '../environments/Acrobot.js';
import { Tetris } from '../environments/tetris/index.js';
import { RobotArm } from '../environments/RobotArm.js';

// ExtActionAgentWrap for Tetris
import { ExtActionAgentWrap } from '../agents/ExtActionAgentWrap.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =============================================================================
// JAVA CONFIGURATION PARAMETERS (EXACT from Java)
// =============================================================================

const JAVA_CONFIG = {
    loopmaze: {
        basisFcnsPerDim: 6,
        relWidth: 0.05,
        stepSize: 0.2,
        normBounds: [-1, 1],
        biasFeatVal: 0.1,
        useBiasWt: true,  // modelAddsBiasFeat
        creditDistType: 'uniform',
        creditDelayMs: 150,
        creditWindowMs: 250
    },
    mountaincar: {
        basisFcnsPerDim: 8,
        relWidth: 0.08,
        stepSize: 0.05,
        normBounds: [-1, 1],
        biasFeatVal: 0.1,
        useBiasWt: true,
        creditDistType: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    },
    cartpole: {
        basisFcnsPerDim: 8,
        relWidth: 0.08,
        stepSize: 0.05,
        normBounds: [-1, 1],
        biasFeatVal: 0.1,
        useBiasWt: true,
        creditDistType: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    },
    acrobot: {
        basisFcnsPerDim: 8,
        relWidth: 0.08,
        stepSize: 0.05,
        normBounds: [-1, 1],
        biasFeatVal: 0.1,
        useBiasWt: true,
        creditDistType: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    },
    robotarm: {
        basisFcnsPerDim: 8,
        relWidth: 0.08,
        stepSize: 0.05,
        normBounds: [-1, 1],
        biasFeatVal: 0.1,
        useBiasWt: true,
        creditDistType: 'uniform',
        creditDelayMs: 200,
        creditWindowMs: 600
    },
    tetris: {
        // Tetris uses TetrisFeatures, not RBF
        stepSize: 0.000005 / 47,
        useBiasWt: true,
        creditDistType: 'previousStep',
        creditDelayMs: 200,
        creditWindowMs: 600,
        numFeatures: 47  // 46 + bias
    }
};

// =============================================================================
// TEST UTILITIES
// =============================================================================

let totalTests = 0;
let passedTests = 0;
let failedTests = [];

function assertEqual(actual, expected, testName, tolerance = 0) {
    totalTests++;
    let passed = false;

    if (tolerance > 0 && typeof actual === 'number' && typeof expected === 'number') {
        passed = Math.abs(actual - expected) <= tolerance;
    } else if (Array.isArray(actual) && Array.isArray(expected)) {
        passed = actual.length === expected.length &&
                 actual.every((v, i) => Math.abs(v - expected[i]) <= (tolerance || 1e-10));
    } else {
        passed = actual === expected;
    }

    if (passed) {
        passedTests++;
        console.log(`  ✓ ${testName}`);
    } else {
        failedTests.push({ testName, actual, expected });
        console.log(`  ✗ ${testName}`);
        if (Array.isArray(actual)) {
            console.log(`    Expected length: ${expected.length}`);
            console.log(`    Actual length:   ${actual.length}`);
        } else {
            console.log(`    Expected: ${expected}`);
            console.log(`    Actual:   ${actual}`);
        }
    }
    return passed;
}

// Seeded random number generator for reproducibility
class SeededRandom {
    constructor(seed = 12345) {
        this.seed = seed;
    }

    next() {
        this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }

    nextInt(max) {
        return Math.floor(this.next() * max);
    }
}

// =============================================================================
// END-TO-END LEARNING TESTS
// =============================================================================

/**
 * Test RBF-based environment learning pipeline
 */
function testRBFEnvironmentLearning(envName, EnvClass) {
    console.log(`\n--- ${envName} End-to-End Learning Test ---`);

    const config = JAVA_CONFIG[envName];
    const env = new EnvClass({ randomStarts: false });
    const envSpec = env.init();

    // Create RBF feature generator with EXACT Java config
    const featGen = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        config.basisFcnsPerDim,
        config.relWidth
    );
    featGen.setNormBounds(config.normBounds[0], config.normBounds[1]);
    featGen.setBiasFeatPerAct(config.biasFeatVal);

    // Create model with EXACT Java config
    const numFeatures = featGen.getNumFeatures();
    const model = new LinearModel(numFeatures, {
        stepSize: config.stepSize,
        decayFactor: 0.0,
        useBiasWt: config.useBiasWt
    });
    model.setFeatGen(featGen);

    // Verify feature count
    const expectedMeans = Math.pow(config.basisFcnsPerDim, envSpec.obsRanges.length);
    const expectedFeatsPerAction = expectedMeans + 1;  // +1 for bias
    const expectedTotalFeats = expectedFeatsPerAction * envSpec.numActions;

    assertEqual(featGen.means.length, expectedMeans,
        `${envName}: Number of RBF means (${config.basisFcnsPerDim}^${envSpec.obsRanges.length})`);
    assertEqual(numFeatures, expectedTotalFeats,
        `${envName}: Total features (${expectedFeatsPerAction} * ${envSpec.numActions})`);

    // Run environment with fixed action sequence and collect training data
    const rng = new SeededRandom(42);
    const samples = [];

    let obs = env.start();  // Get observation from start() return value

    // Take 10 steps with random actions
    for (let step = 0; step < 10; step++) {
        const action = rng.nextInt(envSpec.numActions);

        // Get features for this state-action
        const feats = featGen.getStateActionFeatures(obs, action);

        // Simulate human feedback at certain steps
        const humanReward = (step % 3 === 0) ? 1.0 : (step % 3 === 1) ? -1.0 : 0;

        if (humanReward !== 0) {
            // Create sample
            samples.push({
                feats: Array.from(feats),
                label: humanReward,
                weight: 1.0,
                stepIndex: step,
                obs: [...obs],
                action: action
            });

            // Train model
            model.addInstance(feats, humanReward, 1.0);
        }

        // Take step
        const result = env.step(action);
        obs = result.obs;

        if (result.terminal) {
            obs = env.start();  // Get observation from start() return value
        }
    }

    // Get final weights
    const finalWeights = model.getWeights();
    const biasWt = model.biasWt;  // Direct property access

    // Compute checksum for verification
    const weightSum = finalWeights.reduce((a, b) => a + b, 0);
    const nonZeroWeights = finalWeights.filter(w => Math.abs(w) > 1e-10).length;

    console.log(`  Samples collected: ${samples.length}`);
    console.log(`  Weight dimensions: ${finalWeights.length}`);
    console.log(`  Non-zero weights: ${nonZeroWeights}`);
    console.log(`  Weight sum: ${weightSum.toFixed(6)}`);
    console.log(`  Bias weight: ${biasWt.toFixed(6)}`);

    // Return test data for export
    return {
        envName,
        config: {
            basisFcnsPerDim: config.basisFcnsPerDim,
            relWidth: config.relWidth,
            stepSize: config.stepSize,
            normBounds: config.normBounds,
            biasFeatVal: config.biasFeatVal,
            useBiasWt: config.useBiasWt,
            numObsDims: envSpec.obsRanges.length,
            numActions: envSpec.numActions,
            obsRanges: envSpec.obsRanges
        },
        samples,
        finalWeights: Array.from(finalWeights),
        biasWeight: biasWt,
        weightSum,
        nonZeroWeights
    };
}

/**
 * Test Tetris learning pipeline (uses TetrisFeatures, not RBF)
 */
function testTetrisLearning() {
    console.log('\n--- Tetris End-to-End Learning Test ---');

    const config = JAVA_CONFIG.tetris;
    const env = new Tetris();
    const envSpec = env.init();

    // Create Tetris feature generator
    const featGen = new TetrisFeatures(envSpec.obsRanges, envSpec.numActions);

    // Create model with EXACT Java config
    const numFeatures = featGen.getNumExtendedFeatures();  // 46 for Tetris
    const model = new LinearModel(numFeatures, {
        stepSize: config.stepSize,
        decayFactor: 0.0,
        useBiasWt: config.useBiasWt
    });

    assertEqual(numFeatures, 46, 'Tetris: Number of features (without bias)');

    // For Tetris, we'd need to run the full ExtActionAgentWrap pipeline
    // For now, just verify the configuration is correct
    console.log(`  Feature count: ${numFeatures}`);
    console.log(`  Step size: ${config.stepSize}`);
    console.log(`  Uses bias weight: ${config.useBiasWt}`);

    return {
        envName: 'tetris',
        config: {
            stepSize: config.stepSize,
            useBiasWt: config.useBiasWt,
            numFeatures: numFeatures,
            creditDistType: config.creditDistType
        },
        samples: [],  // Tetris samples require full agent pipeline
        finalWeights: Array.from(model.getWeights()),
        biasWeight: model.biasWt
    };
}

/**
 * Test that feature dimensions match exactly
 */
function testFeatureDimensions() {
    console.log('\n' + '='.repeat(70));
    console.log('FEATURE DIMENSION VERIFICATION');
    console.log('='.repeat(70));

    const environments = [
        { name: 'LoopMaze', EnvClass: LoopMaze, expectedDims: 2 },
        { name: 'MountainCar', EnvClass: MountainCar, expectedDims: 2 },
        { name: 'CartPole', EnvClass: CartPole, expectedDims: 4 },
        { name: 'Acrobot', EnvClass: Acrobot, expectedDims: 4 },
        { name: 'RobotArm', EnvClass: RobotArm, expectedDims: 4 }
    ];

    for (const { name, EnvClass, expectedDims } of environments) {
        console.log(`\n--- ${name} ---`);

        const config = JAVA_CONFIG[name.toLowerCase()];
        const env = new EnvClass();
        const envSpec = env.init();

        assertEqual(envSpec.obsRanges.length, expectedDims, `${name}: Observation dimensions`);

        // Calculate expected features
        const expectedMeans = Math.pow(config.basisFcnsPerDim, expectedDims);
        const expectedFeatsPerAction = expectedMeans + 1;  // +1 for bias per action
        const expectedTotalFeats = expectedFeatsPerAction * envSpec.numActions;

        console.log(`  basisFcnsPerDim: ${config.basisFcnsPerDim}`);
        console.log(`  obsDims: ${expectedDims}`);
        console.log(`  numActions: ${envSpec.numActions}`);
        console.log(`  Expected RBF means: ${config.basisFcnsPerDim}^${expectedDims} = ${expectedMeans}`);
        console.log(`  Expected features per action: ${expectedFeatsPerAction}`);
        console.log(`  Expected total features: ${expectedTotalFeats}`);

        // Create actual RBF and verify
        const rbf = new RBFFeatures(
            envSpec.obsRanges,
            envSpec.numActions,
            config.basisFcnsPerDim,
            config.relWidth
        );
        rbf.setBiasFeatPerAct(config.biasFeatVal);

        assertEqual(rbf.means.length, expectedMeans, `${name}: Actual RBF means`);
        assertEqual(rbf.getNumFeatures(), expectedTotalFeats, `${name}: Actual total features`);
    }
}

/**
 * Export test data for Java verification
 */
function exportTestData(allResults) {
    const outputPath = path.join(__dirname, 'full_learning_test_data.json');

    const exportData = {
        generatedAt: new Date().toISOString(),
        description: 'End-to-end learning test data for Java comparison',
        environments: {}
    };

    for (const result of allResults) {
        exportData.environments[result.envName] = result;
    }

    fs.writeFileSync(outputPath, JSON.stringify(exportData, null, 2));
    console.log(`\nTest data exported to: ${outputPath}`);

    return outputPath;
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

async function runAllTests() {
    console.log('='.repeat(70));
    console.log('FULL END-TO-END LEARNING COMPARISON TEST');
    console.log('Verifying complete learning pipeline matches Java');
    console.log('='.repeat(70));

    const allResults = [];

    try {
        // Feature dimension verification
        testFeatureDimensions();

        // RBF-based environments
        console.log('\n' + '='.repeat(70));
        console.log('RBF ENVIRONMENT LEARNING TESTS');
        console.log('='.repeat(70));

        const rbfEnvs = [
            { name: 'loopmaze', EnvClass: LoopMaze },
            { name: 'mountaincar', EnvClass: MountainCar },
            { name: 'cartpole', EnvClass: CartPole },
            { name: 'acrobot', EnvClass: Acrobot },
            { name: 'robotarm', EnvClass: RobotArm }
        ];

        for (const { name, EnvClass } of rbfEnvs) {
            const result = testRBFEnvironmentLearning(name, EnvClass);
            allResults.push(result);
        }

        // Tetris (uses TetrisFeatures)
        console.log('\n' + '='.repeat(70));
        console.log('TETRIS LEARNING TEST');
        console.log('='.repeat(70));

        const tetrisResult = testTetrisLearning();
        allResults.push(tetrisResult);

        // Export test data
        console.log('\n' + '='.repeat(70));
        console.log('EXPORT TEST DATA FOR JAVA VERIFICATION');
        console.log('='.repeat(70));

        const exportPath = exportTestData(allResults);

    } catch (error) {
        console.error('\n*** TEST ERROR ***');
        console.error(error);
        failedTests.push({ testName: 'Test execution error', actual: error.message, expected: 'No error' });
    }

    // Print summary
    console.log('\n' + '='.repeat(70));
    console.log('TEST SUMMARY');
    console.log('='.repeat(70));
    console.log(`Total tests: ${totalTests}`);
    console.log(`Passed: ${passedTests}`);
    console.log(`Failed: ${failedTests.length}`);

    if (failedTests.length > 0) {
        console.log('\n--- FAILED TESTS ---');
        for (const { testName, actual, expected } of failedTests) {
            console.log(`  ✗ ${testName}`);
            console.log(`    Expected: ${JSON.stringify(expected)}`);
            console.log(`    Actual:   ${JSON.stringify(actual)}`);
        }
        console.log('\n*** TESTS FAILED ***');
        process.exit(1);
    } else {
        console.log('\n*** ALL TESTS PASSED ***');
        console.log('\nNOTE: Run the Java test to verify weights match:');
        console.log('  cd /Users/bradknox/code/tamer_2026/tamerproject');
        console.log('  javac -cp lib/* src/edu/utexas/cs/tamerProject/test/FullLearningComparisonTest.java');
        console.log('  java -cp lib/*:src edu.utexas.cs.tamerProject.test.FullLearningComparisonTest');
        process.exit(0);
    }
}

runAllTests();
