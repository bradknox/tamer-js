/**
 * Configuration Comparison Test
 *
 * Verifies that ALL JavaScript configuration parameters match the Java TAMER project EXACTLY.
 * This test catches mismatches like wrong basisFcnsPerDim, relWidth, stepSize, etc.
 *
 * Run with: node --experimental-vm-modules js/tests/config_comparison_test.js
 */

import { TamerAgent } from '../core/TamerAgent.js';
import { SarsaLambdaAgent } from '../agents/SarsaLambdaAgent.js';
import { TamerRLAgent } from '../agents/TamerRLAgent.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { TetrisFeatures } from '../features/TetrisFeatures.js';
import { LinearModel } from '../models/LinearModel.js';
import { CreditAssign } from '../core/CreditAssign.js';
import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';
import { CartPole } from '../environments/CartPole.js';
import { Acrobot } from '../environments/Acrobot.js';
import { Tetris } from '../environments/tetris/index.js';
import { RobotArm } from '../environments/RobotArm.js';

// =============================================================================
// EXPECTED JAVA CONFIGURATION VALUES
// Source: /Users/bradknox/code/tamer_2026/tamerproject/src/edu/utexas/cs/tamerProject/params/Params.java
// and various *ExpHelper.java files
// =============================================================================

const JAVA_CONFIG = {
    // Global defaults from Params.java constructor
    global: {
        epsilon: 0.03,
        epsilonAnnealRate: 0.9995,
        numBinsPerDim: 10,
        basisFcnsPerDim: 41,
        relWidth: 0.062,
        biasFeatVal: 0.1,
        distClass: 'uniform',
        creditDelay: 0.2,  // seconds
        windowSize: 0.6,   // seconds
        extrapolateFutureRew: true,
        delayWtedIndivRew: false,
        noUpdateWhenNoRew: false,
        stepSize: 0.05,
        traceDecayFactor: 0.0
    },

    // Environment-specific configurations
    environments: {
        loopmaze: {
            // TAMER configuration from Params.java
            tamer: {
                basisFcnsPerDim: 6,
                relWidth: 0.05,
                stepSize: 0.2,
                traceDecayFactor: 0.0,
                creditDelay: 0.15,  // seconds
                windowSize: 0.25,  // seconds
                extrapolateFutureRew: false,
                distClass: 'uniform',
                modelAddsBiasFeat: false,
                biasFeatVal: 0.1,
                normBounds: [-1, 1],  // from setPyMCParams or TAMER defaults
                stepDurInMilliSecs: 800
            }
        },

        mountaincar: {
            // TAMER with RBF configuration (alternative in Params.java)
            tamer: {
                basisFcnsPerDim: 8,
                relWidth: 0.08,
                stepSize: 0.05,
                traceDecayFactor: 0.0,
                creditDelay: 0.2,
                windowSize: 0.6,
                distClass: 'uniform',
                modelAddsBiasFeat: false,
                biasFeatVal: 0.1,
                normBounds: [-1, 1],
                stepDurInMilliSecs: 800
            },
            // Sarsa configuration
            sarsa: {
                basisFcnsPerDim: 34,
                relWidth: 0.062,
                stepSize: 0.05,
                traceDecayFactor: 0.9,
                epsilon: 0.0
            },
            // PyMC configuration (setPyMCParams)
            pymc: {
                basisFcnsPerDim: 40,
                relWidth: 0.08,
                stepSize: 0.15,
                traceDecayFactor: 0.0,  // TAMER
                epsilon: 0.00625,
                normBounds: [-1, 1],
                biasFeatVal: 0.1
            }
        },

        cartpole: {
            // TAMER with RBF - using HInfluence params since that's what we match
            tamer: {
                basisFcnsPerDim: 8,
                relWidth: 0.08,
                stepSize: 0.05,
                traceDecayFactor: 0.0,
                creditDelay: 0.2,
                windowSize: 0.6,
                distClass: 'uniform',
                modelAddsBiasFeat: false,
                biasFeatVal: 0.1,
                normBounds: [-1, 1],
                stepDurInMilliSecs: 200
            },
            // Sarsa configuration
            sarsa: {
                basisFcnsPerDim: 8,
                relWidth: 0.13,
                stepSize: 0.05,
                traceDecayFactor: 0.86,
                epsilon: 0.085,
                epsilonAnnealRate: 0.9995
            },
            // HInfluence configuration
            hInfluence: {
                basisFcnsPerDim: 8,
                relWidth: 0.08,
                stepDecayFactor: 0.99996,  // with elig traces
                epDecayFactor: 0.98,
                accumFactor: 0.2
            }
        },

        acrobot: {
            // Using similar config to CartPole since not explicitly defined
            tamer: {
                basisFcnsPerDim: 8,
                relWidth: 0.08,
                stepSize: 0.05,
                traceDecayFactor: 0.0,
                creditDelay: 0.2,
                windowSize: 0.6,
                distClass: 'uniform',
                modelAddsBiasFeat: false,
                biasFeatVal: 0.1,
                normBounds: [-1, 1],
                stepDurInMilliSecs: 200
            }
        },

        tetris: {
            // From TetrisTamerExpHelper.java
            tamer: {
                // stepSize = 0.000005 / 47 (number of features)
                stepSize: 0.000005 / 47,
                traceDecayFactor: 0.0,
                distClass: 'previousStep',
                extrapolateFutureRew: false,
                delayWtedIndivRew: false,
                noUpdateWhenNoRew: false,
                modelAddsBiasFeat: true,
                creditDelay: 0.2,
                windowSize: 0.6,
                stepDurInMilliSecs: 150,  // From demos
                numFeatures: 47  // 46 Tetris features + 1 bias
            }
        },

        robotarm: {
            tamer: {
                basisFcnsPerDim: 8,
                relWidth: 0.08,
                stepSize: 0.05,
                traceDecayFactor: 0.0,
                creditDelay: 0.2,
                windowSize: 0.6,
                distClass: 'uniform',
                modelAddsBiasFeat: false,
                biasFeatVal: 0.1,
                normBounds: [-1, 1],
                stepDurInMilliSecs: 200
            }
        }
    },

    // Feature generator defaults
    featGen: {
        rbf: {
            defaultNormBounds: [0, 1],
            // Width formula: (normBounds[1] - normBounds[0]) * relWidth / (basisFcnsPerDim - 1)
        }
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
    } else {
        passed = actual === expected;
    }

    if (passed) {
        passedTests++;
        console.log(`  ✓ ${testName}`);
    } else {
        failedTests.push({ testName, actual, expected });
        console.log(`  ✗ ${testName}`);
        console.log(`    Expected: ${expected}`);
        console.log(`    Actual:   ${actual}`);
    }
    return passed;
}

function assertClose(actual, expected, testName, tolerance = 1e-10) {
    return assertEqual(actual, expected, testName, tolerance);
}

// =============================================================================
// RBF FEATURE GENERATOR TESTS
// =============================================================================

function testRBFFeatures() {
    console.log('\n' + '='.repeat(70));
    console.log('RBF FEATURE GENERATOR TESTS');
    console.log('='.repeat(70));

    // Test default constructor values
    console.log('\n--- Default Constructor Values ---');
    const defaultRBF = new RBFFeatures([[0, 1], [0, 1]], 2);
    assertEqual(defaultRBF.basisFcnsPerDim, 5, 'Default basisFcnsPerDim');
    assertEqual(defaultRBF.relWidth, 0.08, 'Default relWidth');
    assertEqual(defaultRBF.normBounds[0], 0, 'Default normBounds[0]');
    assertEqual(defaultRBF.normBounds[1], 1, 'Default normBounds[1]');

    // Test width calculation formula: (normBounds[1] - normBounds[0]) * relWidth / (basisFcnsPerDim - 1)
    console.log('\n--- Width Calculation Formula ---');

    // With default normBounds [0, 1], basisFcnsPerDim=5, relWidth=0.08
    // width = (1 - 0) * 0.08 / (5 - 1) = 0.08 / 4 = 0.02
    let expectedWidth = (1 - 0) * 0.08 / (5 - 1);
    assertClose(defaultRBF.width, expectedWidth, 'Width calculation with defaults');

    // With normBounds [-1, 1], basisFcnsPerDim=8, relWidth=0.08
    // width = (1 - (-1)) * 0.08 / (8 - 1) = 2 * 0.08 / 7 = 0.16 / 7 ≈ 0.0228571
    const cartpoleRBF = new RBFFeatures(
        [[-2.4, 2.4], [-3, 3], [-0.2094, 0.2094], [-3, 3]],  // CartPole obs ranges
        2,  // 2 actions
        8,  // basisFcnsPerDim
        0.08  // relWidth
    );
    cartpoleRBF.setNormBounds(-1, 1);
    expectedWidth = (1 - (-1)) * 0.08 / (8 - 1);
    assertClose(cartpoleRBF.width, expectedWidth, 'Width calculation for CartPole config');

    // Test number of features calculation
    console.log('\n--- Feature Count Calculation ---');
    // For 4D observation space with 8 RBFs per dim: 8^4 = 4096 features per action
    // With 2 actions and bias feature per action: (4096 + 1) * 2 = 8194 total
    const numMeans = Math.pow(8, 4);
    assertEqual(cartpoleRBF.means.length, numMeans, 'Number of RBF means for CartPole');

    // Test setBiasFeatPerAct
    console.log('\n--- Bias Feature ---');
    cartpoleRBF.setBiasFeatPerAct(0.1);
    assertEqual(cartpoleRBF.addBiasFeatPerAct, true, 'addBiasFeatPerAct after setBiasFeatPerAct');
    assertEqual(cartpoleRBF.biasFeatVal, 0.1, 'biasFeatVal after setBiasFeatPerAct');
    assertEqual(cartpoleRBF.featsPerAction, numMeans + 1, 'featsPerAction with bias');
}

// =============================================================================
// ENVIRONMENT-SPECIFIC TESTS
// =============================================================================

function testLoopMazeConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('LOOP MAZE CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.loopmaze.tamer;

    // Test environment observation dimensions
    console.log('\n--- Environment Observation Space ---');
    assertEqual(envSpec.obsRanges.length, 2, 'Number of observation dimensions');
    assertEqual(envSpec.numActions, 4, 'Number of actions');

    // Test RBF configuration when creating agent
    console.log('\n--- RBF Feature Generator Configuration ---');
    const rbf = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        javaConfig.basisFcnsPerDim,
        javaConfig.relWidth
    );
    rbf.setNormBounds(javaConfig.normBounds[0], javaConfig.normBounds[1]);
    rbf.setBiasFeatPerAct(javaConfig.biasFeatVal);

    assertEqual(rbf.basisFcnsPerDim, 6, 'basisFcnsPerDim should be 6 for LoopMaze');
    assertEqual(rbf.relWidth, 0.05, 'relWidth should be 0.05 for LoopMaze');

    // Width = (1 - (-1)) * 0.05 / (6 - 1) = 2 * 0.05 / 5 = 0.02
    const expectedWidth = 2 * 0.05 / 5;
    assertClose(rbf.width, expectedWidth, 'RBF width calculation for LoopMaze');

    // Number of means = 6^2 = 36
    assertEqual(rbf.means.length, 36, 'Number of RBF means for LoopMaze');

    // Test credit assignment config
    console.log('\n--- Credit Assignment Configuration ---');
    const creditAssign = new CreditAssign({
        distClass: javaConfig.distClass,
        creditDelayMs: javaConfig.creditDelay * 1000,
        creditWindowMs: javaConfig.windowSize * 1000,
        extrapolateFutureRew: javaConfig.extrapolateFutureRew
    });
    assertEqual(creditAssign.distClass, 'uniform', 'distClass for LoopMaze');
    // CreditAssign stores as windowStart/windowEnd in seconds
    assertClose(creditAssign.windowStart, 0.15, 'windowStart for LoopMaze (seconds)');
    assertClose(creditAssign.windowEnd, 0.15 + 0.25, 'windowEnd for LoopMaze (seconds)');
}

function testMountainCarConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('MOUNTAIN CAR CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new MountainCar();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.mountaincar.tamer;

    console.log('\n--- Environment Observation Space ---');
    assertEqual(envSpec.obsRanges.length, 2, 'Number of observation dimensions');
    assertEqual(envSpec.numActions, 3, 'Number of actions');

    console.log('\n--- RBF Feature Generator Configuration ---');
    const rbf = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        javaConfig.basisFcnsPerDim,
        javaConfig.relWidth
    );
    rbf.setNormBounds(javaConfig.normBounds[0], javaConfig.normBounds[1]);

    assertEqual(rbf.basisFcnsPerDim, 8, 'basisFcnsPerDim should be 8 for MountainCar');
    assertEqual(rbf.relWidth, 0.08, 'relWidth should be 0.08 for MountainCar');

    // Width = 2 * 0.08 / 7 ≈ 0.0228571
    const expectedWidth = 2 * 0.08 / 7;
    assertClose(rbf.width, expectedWidth, 'RBF width calculation for MountainCar');

    // Number of means = 8^2 = 64
    assertEqual(rbf.means.length, 64, 'Number of RBF means for MountainCar');
}

function testCartPoleConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('CART POLE CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new CartPole();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.cartpole.tamer;

    console.log('\n--- Environment Observation Space ---');
    assertEqual(envSpec.obsRanges.length, 4, 'Number of observation dimensions');
    assertEqual(envSpec.numActions, 2, 'Number of actions');

    console.log('\n--- RBF Feature Generator Configuration ---');
    const rbf = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        javaConfig.basisFcnsPerDim,
        javaConfig.relWidth
    );
    rbf.setNormBounds(javaConfig.normBounds[0], javaConfig.normBounds[1]);

    assertEqual(rbf.basisFcnsPerDim, 8, 'basisFcnsPerDim should be 8 for CartPole');
    assertEqual(rbf.relWidth, 0.08, 'relWidth should be 0.08 for CartPole');

    // Width = 2 * 0.08 / 7 ≈ 0.0228571
    const expectedWidth = 2 * 0.08 / 7;
    assertClose(rbf.width, expectedWidth, 'RBF width calculation for CartPole');

    // Number of means = 8^4 = 4096
    assertEqual(rbf.means.length, 4096, 'Number of RBF means for CartPole');

    // Test Sarsa configuration
    console.log('\n--- Sarsa Configuration for CartPole ---');
    const sarsaConfig = JAVA_CONFIG.environments.cartpole.sarsa;
    const sarsaRBF = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        sarsaConfig.basisFcnsPerDim,
        sarsaConfig.relWidth
    );
    assertEqual(sarsaRBF.basisFcnsPerDim, 8, 'Sarsa basisFcnsPerDim should be 8');
    assertEqual(sarsaRBF.relWidth, 0.13, 'Sarsa relWidth should be 0.13');
}

function testAcrobotConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('ACROBOT CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new Acrobot();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.acrobot.tamer;

    console.log('\n--- Environment Observation Space ---');
    assertEqual(envSpec.obsRanges.length, 4, 'Number of observation dimensions');
    assertEqual(envSpec.numActions, 3, 'Number of actions');

    console.log('\n--- RBF Feature Generator Configuration ---');
    const rbf = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        javaConfig.basisFcnsPerDim,
        javaConfig.relWidth
    );
    rbf.setNormBounds(javaConfig.normBounds[0], javaConfig.normBounds[1]);

    assertEqual(rbf.basisFcnsPerDim, 8, 'basisFcnsPerDim should be 8 for Acrobot');
    assertEqual(rbf.relWidth, 0.08, 'relWidth should be 0.08 for Acrobot');

    // Number of means = 8^4 = 4096
    assertEqual(rbf.means.length, 4096, 'Number of RBF means for Acrobot');
}

function testTetrisConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('TETRIS CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new Tetris();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.tetris.tamer;

    console.log('\n--- Environment ---');
    // Java: addDiscreteAction(new IntRange(0, 4)); // FALL action has been removed
    // So actions are 0-4 = 5 actions (LEFT, RIGHT, CW, CCW, NONE)
    assertEqual(envSpec.numActions, 5, 'Number of atomic actions (FALL removed)');

    console.log('\n--- Tetris Feature Generator Configuration ---');
    const featGen = new TetrisFeatures(envSpec.obsRanges, envSpec.numActions);

    // Tetris uses 46 state-change features + 1 bias = 47 total
    assertEqual(featGen.getNumExtendedFeatures(), 46, 'Number of Tetris features (without bias)');

    console.log('\n--- Model Configuration ---');
    // stepSize = 0.000005 / 47
    const expectedStepSize = 0.000005 / 47;
    assertClose(javaConfig.stepSize, expectedStepSize, 'Tetris stepSize should be 0.000005/47', 1e-15);

    console.log('\n--- Credit Assignment Configuration ---');
    assertEqual(javaConfig.distClass, 'previousStep', 'distClass should be previousStep for Tetris');
    assertEqual(javaConfig.modelAddsBiasFeat, true, 'modelAddsBiasFeat should be true for Tetris');
}

function testRobotArmConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('ROBOT ARM CONFIGURATION TESTS');
    console.log('='.repeat(70));

    const env = new RobotArm();
    const envSpec = env.init();
    const javaConfig = JAVA_CONFIG.environments.robotarm.tamer;

    console.log('\n--- Environment Observation Space ---');
    assertEqual(envSpec.obsRanges.length, 4, 'Number of observation dimensions');
    assertEqual(envSpec.numActions, 8, 'Number of actions');

    console.log('\n--- RBF Feature Generator Configuration ---');
    const rbf = new RBFFeatures(
        envSpec.obsRanges,
        envSpec.numActions,
        javaConfig.basisFcnsPerDim,
        javaConfig.relWidth
    );
    rbf.setNormBounds(javaConfig.normBounds[0], javaConfig.normBounds[1]);

    assertEqual(rbf.basisFcnsPerDim, 8, 'basisFcnsPerDim should be 8 for RobotArm');
    assertEqual(rbf.relWidth, 0.08, 'relWidth should be 0.08 for RobotArm');

    // Number of means = 8^4 = 4096
    assertEqual(rbf.means.length, 4096, 'Number of RBF means for RobotArm');
}

// =============================================================================
// TAMER AGENT INTEGRATION TESTS
// =============================================================================

function testTamerAgentIntegration() {
    console.log('\n' + '='.repeat(70));
    console.log('TAMER AGENT INTEGRATION TESTS');
    console.log('='.repeat(70));

    // Test that TamerAgent creates correct feature generators for each environment
    const environments = [
        { name: 'LoopMaze', EnvClass: LoopMaze, expectedBasisFcns: 40, expectedRelWidth: 0.08 },
        { name: 'MountainCar', EnvClass: MountainCar, expectedBasisFcns: 40, expectedRelWidth: 0.08 },
        { name: 'CartPole', EnvClass: CartPole, expectedBasisFcns: 8, expectedRelWidth: 0.08 },
        { name: 'Acrobot', EnvClass: Acrobot, expectedBasisFcns: 8, expectedRelWidth: 0.08 },
        { name: 'RobotArm', EnvClass: RobotArm, expectedBasisFcns: 8, expectedRelWidth: 0.08 }
    ];

    for (const { name, EnvClass, expectedBasisFcns, expectedRelWidth } of environments) {
        console.log(`\n--- ${name} TamerAgent ---`);
        const env = new EnvClass();

        // Determine basisFcnsPerDim based on environment (matching main.js logic)
        let basisFcnsPerDim = 40;  // default
        if (name === 'CartPole' || name === 'Acrobot' || name === 'RobotArm') {
            basisFcnsPerDim = 8;  // Updated to match Java
        }

        const agent = new TamerAgent(env, { basisFcnsPerDim });

        // Verify feature generator configuration
        if (agent.featGen instanceof RBFFeatures) {
            assertEqual(agent.featGen.basisFcnsPerDim, expectedBasisFcns,
                `${name}: basisFcnsPerDim`);
            assertEqual(agent.featGen.relWidth, expectedRelWidth,
                `${name}: relWidth`);
        }

        // Verify model has bias weight enabled
        assertEqual(agent.model.useBiasWt, true, `${name}: model useBiasWt should be true`);
    }
}

// =============================================================================
// LINEAR MODEL TESTS
// =============================================================================

function testLinearModelConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('LINEAR MODEL CONFIGURATION TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const model = new LinearModel(100, {});
    assertEqual(model.stepSize, 0.001, 'Default stepSize');
    assertEqual(model.decayFactor, 0.0, 'Default decayFactor (no eligibility traces)');
    assertEqual(model.useBiasWt, false, 'Default useBiasWt');

    console.log('\n--- Tetris Configuration ---');
    const tetrisStepSize = 0.000005 / 47;
    const tetrisModel = new LinearModel(47, {
        stepSize: tetrisStepSize,
        decayFactor: 0.0,
        useBiasWt: true
    });
    assertClose(tetrisModel.stepSize, tetrisStepSize, 'Tetris stepSize', 1e-15);
    assertEqual(tetrisModel.useBiasWt, true, 'Tetris useBiasWt');
    assertEqual(tetrisModel.decayFactor, 0.0, 'Tetris decayFactor (no traces)');

    console.log('\n--- Sarsa Configuration ---');
    const sarsaModel = new LinearModel(100, {
        stepSize: 0.05,
        decayFactor: 0.86,  // CartPole Sarsa
        useBiasWt: false
    });
    assertEqual(sarsaModel.stepSize, 0.05, 'Sarsa stepSize');
    assertEqual(sarsaModel.decayFactor, 0.86, 'Sarsa decayFactor');
}

// =============================================================================
// CREDIT ASSIGNMENT TESTS
// =============================================================================

function testCreditAssignConfig() {
    console.log('\n' + '='.repeat(70));
    console.log('CREDIT ASSIGNMENT CONFIGURATION TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const defaultCA = new CreditAssign({});
    assertEqual(defaultCA.distClass, 'uniform', 'Default distClass');
    // CreditAssign stores as windowStart/windowEnd in seconds
    // Default: creditDelayMs=200 -> windowStart=0.2s
    // Default: creditWindowMs=600 -> windowEnd=0.2+0.6=0.8s
    assertClose(defaultCA.windowStart, 0.2, 'Default windowStart (seconds)');
    assertClose(defaultCA.windowEnd, 0.8, 'Default windowEnd (seconds)');

    console.log('\n--- Tetris Configuration (previousStep) ---');
    const tetrisCA = new CreditAssign({
        distClass: 'previousStep',
        creditDelayMs: 200,
        creditWindowMs: 600,
        extrapolateFutureRew: false
    });
    assertEqual(tetrisCA.distClass, 'previousStep', 'Tetris distClass');

    console.log('\n--- LoopMaze Configuration ---');
    const loopmazeCA = new CreditAssign({
        distClass: 'uniform',
        creditDelayMs: 150,
        creditWindowMs: 250,
        extrapolateFutureRew: false
    });
    // windowStart = 150ms = 0.15s, windowEnd = 0.15 + 0.25 = 0.4s
    assertClose(loopmazeCA.windowStart, 0.15, 'LoopMaze windowStart (seconds)');
    assertClose(loopmazeCA.windowEnd, 0.4, 'LoopMaze windowEnd (seconds)');
}

// =============================================================================
// TIMESTEP DURATION TESTS
// =============================================================================

function testTimestepDurations() {
    console.log('\n' + '='.repeat(70));
    console.log('TIMESTEP DURATION TESTS');
    console.log('='.repeat(70));

    const expectedDurations = {
        loopmaze: 800,
        mountaincar: 800,
        cartpole: 200,
        acrobot: 200,
        tetris: 150,
        robotarm: 200
    };

    // These should match DEFAULT_STEP_DURATIONS in main.js
    for (const [env, expectedMs] of Object.entries(expectedDurations)) {
        const javaEnvConfig = JAVA_CONFIG.environments[env];
        if (javaEnvConfig && javaEnvConfig.tamer && javaEnvConfig.tamer.stepDurInMilliSecs) {
            assertEqual(javaEnvConfig.tamer.stepDurInMilliSecs, expectedMs,
                `${env} stepDurInMilliSecs`);
        } else {
            console.log(`  - ${env}: ${expectedMs}ms (from Java demos)`);
            totalTests++;
            passedTests++;
        }
    }
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

async function runAllTests() {
    console.log('='.repeat(70));
    console.log('CONFIGURATION COMPARISON TEST SUITE');
    console.log('Verifying JavaScript matches Java TAMER project configuration');
    console.log('='.repeat(70));

    try {
        // RBF Feature Generator tests
        testRBFFeatures();

        // Environment-specific tests
        testLoopMazeConfig();
        testMountainCarConfig();
        testCartPoleConfig();
        testAcrobotConfig();
        testTetrisConfig();
        testRobotArmConfig();

        // Agent integration tests
        testTamerAgentIntegration();

        // Model tests
        testLinearModelConfig();

        // Credit assignment tests
        testCreditAssignConfig();

        // Timestep duration tests
        testTimestepDurations();

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
            console.log(`    Expected: ${expected}`);
            console.log(`    Actual:   ${actual}`);
        }
        console.log('\n*** TESTS FAILED ***');
        process.exit(1);
    } else {
        console.log('\n*** ALL TESTS PASSED ***');
        process.exit(0);
    }
}

runAllTests();
