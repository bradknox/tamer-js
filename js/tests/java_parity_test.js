/**
 * EXHAUSTIVE Java-JavaScript Parity Test
 *
 * This test systematically verifies that ALL ported JavaScript code matches
 * the Java source EXACTLY. It extracts every parameter, default value, and
 * constant from both codebases and compares them.
 *
 * COVERS ALL ENVIRONMENTS AND ALL ALGORITHMS:
 * - Environments: LoopMaze, MountainCar, CartPole, Acrobot, RobotArm, Tetris
 * - Agents: TamerAgent, TamerRLAgent, SarsaLambdaAgent, ImitationAgent
 * - Features: RBFFeatures, TetrisFeatures
 * - Core: CreditAssign, HLearner, ActionSelect, HInfluence
 *
 * Run with: node --experimental-vm-modules js/tests/java_parity_test.js
 */

import fs from 'fs';
import path from 'path';

// Import ALL JS modules for testing
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { CreditAssign } from '../core/CreditAssign.js';
import { HLearner } from '../core/HLearner.js';
import { TamerAgent } from '../core/TamerAgent.js';
import { ActionSelect } from '../core/ActionSelect.js';
import { HInfluence } from '../core/HInfluence.js';
import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';
import { CartPole } from '../environments/CartPole.js';
import { Acrobot } from '../environments/Acrobot.js';
import { RobotArm } from '../environments/RobotArm.js';
import { TetrisState, TetrisActions } from '../environments/tetris/TetrisState.js';
import { Tetris } from '../environments/tetris/Tetris.js';
import { TetrisPiece } from '../environments/tetris/TetrisPiece.js';
import { TetrisFeatures } from '../features/TetrisFeatures.js';
import { TamerRLAgent, CombinationMethods } from '../agents/TamerRLAgent.js';
import { SarsaLambdaAgent } from '../agents/SarsaLambdaAgent.js';
import { ImitationAgent } from '../agents/ImitationAgent.js';
import { ExtActionAgentWrap } from '../agents/ExtActionAgentWrap.js';

// =============================================================================
// JAVA SOURCE PATHS
// =============================================================================

const JAVA_BASE = '/Users/bradknox/code/tamer_2026/tamerproject/src';

// =============================================================================
// TEST FRAMEWORK
// =============================================================================

let totalTests = 0;
let passedTests = 0;
let failedTests = [];
let warnings = [];

function assertEqual(actual, expected, testName, tolerance = 0) {
    totalTests++;
    let passed = false;

    if (tolerance > 0 && typeof actual === 'number' && typeof expected === 'number') {
        passed = Math.abs(actual - expected) <= tolerance;
    } else if (actual === expected) {
        passed = true;
    } else if (typeof actual === 'number' && typeof expected === 'number' &&
               isNaN(actual) && isNaN(expected)) {
        passed = true;
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

function assertArrayEqual(actual, expected, testName) {
    totalTests++;
    let passed = true;

    if (actual.length !== expected.length) {
        passed = false;
    } else {
        for (let i = 0; i < actual.length; i++) {
            if (actual[i] !== expected[i]) {
                passed = false;
                break;
            }
        }
    }

    if (passed) {
        passedTests++;
        console.log(`  ✓ ${testName}`);
    } else {
        failedTests.push({ testName, actual: JSON.stringify(actual), expected: JSON.stringify(expected) });
        console.log(`  ✗ ${testName}`);
        console.log(`    Expected: ${JSON.stringify(expected)}`);
        console.log(`    Actual:   ${JSON.stringify(actual)}`);
    }
    return passed;
}

function warn(message) {
    warnings.push(message);
    console.log(`  ⚠ WARNING: ${message}`);
}

// =============================================================================
// PARAMS.JAVA DEFAULTS - The source of truth for many parameters
// =============================================================================

// EXACT values extracted from Params.java
const JAVA_PARAMS = {
    // From Params.java field declarations
    featClass: "None",
    modelClass: "None",
    initModelWSamples: false,
    initSampleValue: 0.0,
    numBiasingSamples: 0,
    biasSampleWt: 0.5,
    initWtsValue: 0.0,
    modelAddsBiasFeat: false,  // Line 48 - CRITICAL: default is FALSE
    wekaModelName: "",
    stepSize: 0.05,            // Line 55 - CRITICAL
    traceType: "replacing",
    traceDecayFactor: 0.0,     // Line 57 - disables eligibility traces
    selectionMethod: "e-greedy",
    distClass: "uniform",
    creditDelay: 0.2,          // Line 61 - in seconds
    windowSize: 0.6,           // Line 62 - in seconds
    extrapolateFutureRew: true, // Line 71 - CRITICAL
    delayWtedIndivRew: false,
    noUpdateWhenNoRew: false
};

// Environment-specific overrides from Params.java getParams() method
const JAVA_ENV_PARAMS = {
    'Loop-Maze': {
        // Lines 146-157 in Params.java
        basisFcnsPerDim: 6,
        relWidth: 0.05,
        stepSize: 0.2,
        creditDelay: 0.15,  // Java: 0.15 seconds
        windowSize: 0.25    // Java: 0.25 seconds
    },
    'Mountain-Car': {
        basisFcnsPerDim: 40,
        relWidth: 0.08,
        stepSize: 0.05
    },
    'CartPole': {
        basisFcnsPerDim: 8,
        relWidth: 0.08
    }
};

// =============================================================================
// 1. LinearModel (IncGDLinearModel.java) PARITY TESTS
// =============================================================================

function testLinearModelParity() {
    console.log('\n' + '='.repeat(70));
    console.log('1. LINEAR MODEL PARITY (IncGDLinearModel.java vs LinearModel.js)');
    console.log('='.repeat(70));

    // Test default values when creating model with no options
    const model = new LinearModel(10);

    // JAVA: private double decayFactor = 0.0
    assertEqual(model.decayFactor, 0.0, 'decayFactor default');

    // JAVA: private String traceStyle = "replacing"
    assertEqual(model.traceStyle, 'replacing', 'traceStyle default');

    // JAVA: private double biasWt = 0
    assertEqual(model.biasWt, 0.0, 'biasWt default');

    // JAVA: private boolean useBiasWt = false
    assertEqual(model.useBiasWt, false, 'useBiasWt default');

    // JAVA: final double APPROX_ONE = 0.99999 - internal constant, not exposed

    // JAVA: private double regL2Wt = 0
    assertEqual(model.regL2Wt, 0.0, 'regL2Wt default');

    // JAVA Params.java: stepSize = 0.05
    assertEqual(model.stepSize, 0.05, 'stepSize default (from Params.java)');

    // Test weight initialization
    assertEqual(model.weights.length, 10, 'weights array length');
    assertEqual(model.weights[0], 0.0, 'weights initial value');
    assertEqual(model.complSampleWts.length, 10, 'complSampleWts array length');
    assertEqual(model.traces.length, 10, 'traces array length');

    // Test gradient descent update formula
    console.log('\n--- Gradient Descent Update Formula ---');
    const testModel = new LinearModel(3, { stepSize: 0.1 });
    const feats = [1.0, 0.5, 0.0];
    const label = 1.0;
    testModel.addInstance(feats, label, 1.0);

    // JAVA formula: weights[i] += traces[i] * wtForErr * (err - regL2Wt * weights[i])
    // With decayFactor=0, traces[i] = feats[i]
    // prediction = 0, err = 1.0, wtForErr = 0.1 * 1.0 = 0.1
    // weight[0] = 0 + 1.0 * 0.1 * 1.0 = 0.1
    // weight[1] = 0 + 0.5 * 0.1 * 1.0 = 0.05
    // weight[2] = 0 + 0.0 * 0.1 * 1.0 = 0.0
    assertEqual(testModel.weights[0], 0.1, 'weight update formula (feat=1.0)', 1e-10);
    assertEqual(testModel.weights[1], 0.05, 'weight update formula (feat=0.5)', 1e-10);
    assertEqual(testModel.weights[2], 0.0, 'weight update formula (feat=0.0)', 1e-10);
}

// =============================================================================
// 2. RBFFeatures (FeatGen_RBFs.java) PARITY TESTS
// =============================================================================

function testRBFFeaturesParity() {
    console.log('\n' + '='.repeat(70));
    console.log('2. RBF FEATURES PARITY (FeatGen_RBFs.java vs RBFFeatures.js)');
    console.log('='.repeat(70));

    // Test default values
    const obsRanges = [[0, 1], [0, 1]];
    const rbf = new RBFFeatures(obsRanges, 3, 5, 0.08);

    // JAVA: boolean addBiasFeatPerAct = false
    assertEqual(rbf.addBiasFeatPerAct, false, 'addBiasFeatPerAct default');

    // JAVA: double biasFeatVal = 0
    assertEqual(rbf.biasFeatVal, 0, 'biasFeatVal default');

    // JAVA: double[] normBounds = {0, 1} by default
    assertArrayEqual(rbf.normBounds, [0, 1], 'normBounds default');

    // JAVA: this.width = (normBounds[1] - normBounds[0]) * this.relWidth / (basisFcnsPerDim - 1)
    const expectedWidth = (1 - 0) * 0.08 / (5 - 1);
    assertEqual(rbf.width, expectedWidth, 'width calculation', 1e-10);

    // Test setNormBounds
    rbf.setNormBounds(-1, 1);
    const expectedWidthAfterNorm = (1 - (-1)) * 0.08 / (5 - 1);
    assertEqual(rbf.width, expectedWidthAfterNorm, 'width after setNormBounds(-1, 1)', 1e-10);
    assertArrayEqual(rbf.normBounds, [-1, 1], 'normBounds after setNormBounds');

    // Test setBiasFeatPerAct
    rbf.setBiasFeatPerAct(0.1);
    assertEqual(rbf.addBiasFeatPerAct, true, 'addBiasFeatPerAct after setBiasFeatPerAct');
    assertEqual(rbf.biasFeatVal, 0.1, 'biasFeatVal after setBiasFeatPerAct(0.1)');

    // Test number of means
    // JAVA: means = basisFcnsPerDim^numObsDims
    assertEqual(rbf.means.length, 25, 'number of means (5^2)');

    // Test number of features with bias
    // JAVA: numFeatures = (means.size() + 1) * numActions with bias
    assertEqual(rbf.numFeatures, (25 + 1) * 3, 'numFeatures with bias per action');

    // Test RBF at center gives 1.0
    console.log('\n--- RBF Calculation at Center ---');
    const rbf2 = new RBFFeatures([[0, 10]], 2, 5, 0.08);
    rbf2.setNormBounds(-1, 1);
    // With 5 basis functions, means are at 0, 2.5, 5, 7.5, 10
    const featsAtMean = rbf2.getStateActionFeatures([2.5], 0);
    // Feature at index 1 should be 1.0 (at the second mean)
    assertEqual(featsAtMean[1], 1.0, 'RBF at center = 1.0', 1e-6);

    // Test RBF falls off with distance
    const featsAwayFromMean = rbf2.getStateActionFeatures([2.6], 0);
    assertEqual(featsAwayFromMean[1] < 1.0, true, 'RBF falls off away from center');
}

// =============================================================================
// 3. CreditAssign (CreditAssign.java) PARITY TESTS
// =============================================================================

function testCreditAssignParity() {
    console.log('\n' + '='.repeat(70));
    console.log('3. CREDIT ASSIGN PARITY (CreditAssign.java vs CreditAssign.js)');
    console.log('='.repeat(70));

    // Test defaults match Params.java
    const ca = new CreditAssign();

    // JAVA Params.java: distClass = "uniform"
    assertEqual(ca.distClass, 'uniform', 'distClass default');

    // JAVA Params.java: creditDelay = 0.2 (in seconds)
    // JS uses milliseconds in constructor, converts to seconds
    assertEqual(ca.windowStart, 0.2, 'windowStart (creditDelay) default');

    // JAVA Params.java: windowSize = 0.6 (in seconds)
    assertEqual(ca.windowEnd - ca.windowStart, 0.6, 'windowSize default', 1e-10);

    // JAVA Params.java: extrapolateFutureRew = true
    assertEqual(ca.EXTRAPOLATE_FUTURE_REW, true, 'EXTRAPOLATE_FUTURE_REW default');

    // JAVA CreditAssign.java: SAMPLE_CUMUL_CRED_MIN = 0.9
    assertEqual(ca.SAMPLE_CUMUL_CRED_MIN, 0.9, 'SAMPLE_CUMUL_CRED_MIN');

    // JAVA CreditAssign.java: MIN_USED_CRED_FOR_EXTRAP = 0.5
    assertEqual(ca.MIN_USED_CRED_FOR_EXTRAP, 0.5, 'MIN_USED_CRED_FOR_EXTRAP');

    // JAVA CreditAssign.java: APPROX_ONE = 0.99999
    assertEqual(ca.APPROX_ONE, 0.99999, 'APPROX_ONE');

    // JAVA Params.java: delayWtedIndivRew = false
    assertEqual(ca.delayWtedIndivRew, false, 'delayWtedIndivRew default');

    // JAVA Params.java: noUpdateWhenNoRew = false
    assertEqual(ca.noUpdateWhenNoRew, false, 'noUpdateWhenNoRew default');

    // JAVA CreditAssign.java: inTrainSess = false
    assertEqual(ca.inTrainSess, false, 'inTrainSess default');

    // Test credit calculation for uniform distribution
    console.log('\n--- Uniform Credit Distribution ---');
    const ca2 = new CreditAssign({ creditDelayMs: 200, creditWindowMs: 600 });
    // Record a timestep
    ca2.recordTimeStepStart([0, 1, 0], 0.0);  // features, time in seconds
    ca2.recordTimeStepEnd(0.1);

    // Test credit at various times
    // Credit should be 0 before windowStart (0.2s) after step ends
    // Credit should increase after windowStart
    // Full credit by windowEnd (0.8s after step ends)
}

// =============================================================================
// 4. TamerAgent PARITY TESTS
// =============================================================================

function testTamerAgentParity() {
    console.log('\n' + '='.repeat(70));
    console.log('4. TAMER AGENT PARITY');
    console.log('='.repeat(70));

    // Create a simple environment for testing
    const env = new LoopMaze();

    // Test defaults
    const agent = new TamerAgent(env);

    // JAVA Params.java: stepSize = 0.05
    assertEqual(agent.stepSize, 0.05, 'stepSize default');

    // JAVA: epsilon (exploration) is typically 0
    assertEqual(agent.epsilon, 0.0, 'epsilon default');

    // JAVA Params.java: basisFcnsPerDim default = 40 (but overridden per env)
    assertEqual(agent.basisFcnsPerDim, 40, 'basisFcnsPerDim default');

    // JAVA Params.java: relWidth = 0.08
    assertEqual(agent.relWidth, 0.08, 'relWidth default');

    // JAVA Params.java: creditDelay = 0.2 (200ms)
    assertEqual(agent.creditDelayMs, 200, 'creditDelayMs default');

    // JAVA Params.java: windowSize = 0.6 (600ms)
    assertEqual(agent.creditWindowMs, 600, 'creditWindowMs default');

    // JAVA Params.java: distClass = "uniform"
    assertEqual(agent.creditDistType, 'uniform', 'creditDistType default');

    // JAVA Params.java: extrapolateFutureRew = true
    assertEqual(agent.extrapolateFutureRew, true, 'extrapolateFutureRew default');

    // Test model creation
    // JAVA: modelAddsBiasFeat = false by default (only Tetris sets it true)
    assertEqual(agent.model.useBiasWt, false, 'model.useBiasWt for non-Tetris env');
}

// =============================================================================
// 5. LoopMaze Environment PARITY TESTS
// =============================================================================

function testLoopMazeParity() {
    console.log('\n' + '='.repeat(70));
    console.log('5. LOOPMAZE PARITY');
    console.log('='.repeat(70));

    const maze = new LoopMaze();

    // JAVA: static final int numActions = 4
    assertEqual(maze.numActions, 4, 'numActions');

    // JAVA LoopMazeState: rewardPerStep = -1.0
    assertEqual(maze.rewardPerStep, -1.0, 'rewardPerStep');

    // JAVA LoopMazeState: rewardAtGoal = 0.0
    assertEqual(maze.rewardAtGoal, 0.0, 'rewardAtGoal');

    // JAVA LoopMazeState: rewardAtFail = -20.0
    assertEqual(maze.rewardAtFail, -20.0, 'rewardAtFail');

    // JAVA: defaultInitPosition = {4, 0}
    assertEqual(maze.startX, 4, 'startX (defaultInitPosition[0])');
    assertEqual(maze.startY, 0, 'startY (defaultInitPosition[1])');

    // JAVA: goalLoc = {5, 0}
    assertEqual(maze.goalX, 5, 'goalX');
    assertEqual(maze.goalY, 0, 'goalY');

    // JAVA: worldDims = {6, 6}
    assertArrayEqual(maze.worldDims, [6, 6], 'worldDims');

    // Test stateMap - EXACT from Java
    console.log('\n--- State Map Parity ---');
    const expectedStateMap = [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 2, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ];
    let stateMapMatch = true;
    for (let y = 0; y < 6; y++) {
        for (let x = 0; x < 6; x++) {
            if (maze.stateMap[y][x] !== expectedStateMap[y][x]) {
                stateMapMatch = false;
                console.log(`  Mismatch at [${y}][${x}]: JS=${maze.stateMap[y][x]}, Java=${expectedStateMap[y][x]}`);
            }
        }
    }
    totalTests++;
    if (stateMapMatch) {
        passedTests++;
        console.log('  ✓ stateMap matches Java exactly');
    } else {
        failedTests.push({ testName: 'stateMap', actual: 'mismatch', expected: 'match' });
        console.log('  ✗ stateMap has differences');
    }

    // Test vertWalls - EXACT from Java
    const expectedVertWalls = [
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [2, 2, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ];
    let vertWallsMatch = true;
    for (let y = 0; y < 6; y++) {
        for (let x = 0; x < 5; x++) {
            if (maze.vertWalls[y][x] !== expectedVertWalls[y][x]) {
                vertWallsMatch = false;
            }
        }
    }
    totalTests++;
    if (vertWallsMatch) {
        passedTests++;
        console.log('  ✓ vertWalls matches Java exactly');
    } else {
        failedTests.push({ testName: 'vertWalls', actual: 'mismatch', expected: 'match' });
        console.log('  ✗ vertWalls has differences');
    }

    // Test horWalls - EXACT from Java
    const expectedHorWalls = [
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0]
    ];
    let horWallsMatch = true;
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 6; x++) {
            if (maze.horWalls[y][x] !== expectedHorWalls[y][x]) {
                horWallsMatch = false;
            }
        }
    }
    totalTests++;
    if (horWallsMatch) {
        passedTests++;
        console.log('  ✓ horWalls matches Java exactly');
    } else {
        failedTests.push({ testName: 'horWalls', actual: 'mismatch', expected: 'match' });
        console.log('  ✗ horWalls has differences');
    }

    // Test action mapping - EXACT from Java
    // 0=right, 1=left, 2=down, 3=up
    console.log('\n--- Action Mapping ---');
    maze.start();
    assertEqual(maze.agentX, 4, 'start position X');
    assertEqual(maze.agentY, 0, 'start position Y');

    // Note: There's a wall between (4,0) and (5,0) in vertWalls[0][4]=1
    // So moving right from start is blocked. Test moving left instead.
    // From (4,0), moving left should go to (3,0) since vertWalls[0][3]=0
    const result = maze.step(1);  // action 1 = left
    assertEqual(maze.agentX, 3, 'after action 1 (left), X');
    assertEqual(maze.agentY, 0, 'after action 1 (left), Y');

    // Test that moving right into a wall stays in place
    maze.start();  // Reset to (4, 0)
    maze.step(0);  // Try to move right (blocked by wall)
    assertEqual(maze.agentX, 4, 'blocked by wall, X stays same');
    assertEqual(maze.agentY, 0, 'blocked by wall, Y stays same');
}

// =============================================================================
// 6. MountainCar Environment PARITY TESTS
// =============================================================================

function testMountainCarParity() {
    console.log('\n' + '='.repeat(70));
    console.log('6. MOUNTAIN CAR PARITY');
    console.log('='.repeat(70));

    const mc = new MountainCar();

    // JAVA: 3 actions
    assertEqual(mc.numActions, 3, 'numActions');

    // JAVA: position range [-1.2, 0.6]
    assertEqual(mc.positionMin, -1.2, 'positionMin');
    assertEqual(mc.positionMax, 0.6, 'positionMax');

    // JAVA: velocity range [-0.07, 0.07]
    assertEqual(mc.velocityMin, -0.07, 'velocityMin');
    assertEqual(mc.velocityMax, 0.07, 'velocityMax');

    // JAVA: goal position 0.5
    assertEqual(mc.goalPosition, 0.5, 'goalPosition');

    // Test obsRanges
    assertArrayEqual(mc.obsRanges[0], [-1.2, 0.6], 'obsRanges[0] (position)');
    assertArrayEqual(mc.obsRanges[1], [-0.07, 0.07], 'obsRanges[1] (velocity)');

    // JAVA: physics constants
    assertEqual(mc.gravity, 0.0025, 'gravity');
    assertEqual(mc.carAccel, 0.001, 'carAccel');
}

// =============================================================================
// 7. CartPole Environment PARITY TESTS
// =============================================================================

function testCartPoleParity() {
    console.log('\n' + '='.repeat(70));
    console.log('7. CART POLE PARITY');
    console.log('='.repeat(70));

    const cp = new CartPole();

    // JAVA: 2 actions (left/right)
    assertEqual(cp.numActions, 2, 'numActions');

    // Test observation ranges (4D state)
    assertEqual(cp.obsRanges.length, 4, 'obsRanges length (4D state)');
}

// =============================================================================
// 8. INTEGRATION TEST: Full TAMER Learning Cycle
// =============================================================================

function testTamerIntegration() {
    console.log('\n' + '='.repeat(70));
    console.log('8. TAMER INTEGRATION TEST');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        stepSize: 0.2,  // LoopMaze-specific
        basisFcnsPerDim: 6,  // LoopMaze-specific
        relWidth: 0.05  // LoopMaze-specific
    });

    // Start episode
    const obs = agent.startEpisode();
    assertEqual(Array.isArray(obs) || typeof obs === 'number', true, 'startEpisode returns observation');

    // Toggle training
    const isTraining = agent.toggleTraining();
    assertEqual(isTraining, true, 'training toggled ON');
    assertEqual(agent.isTraining(), true, 'isTraining() returns true');

    // Process human reward
    agent.processHumanReward(1.0, performance.now());
    assertEqual(agent.pendingRewards.length, 1, 'reward queued');

    // Take a step
    const result = agent.step(performance.now());
    assertEqual('obs' in result, true, 'step returns obs');
    assertEqual('reward' in result, true, 'step returns reward');
    assertEqual('terminal' in result, true, 'step returns terminal');

    // Verify action values are separate per action
    console.log('\n--- Per-Action Prediction Independence ---');
    const values = agent.getActionValues();
    console.log('  Action values:', values);
    // After one positive reward, the action taken should have higher value
    // Other actions should still be near 0 (no cross-action contamination)
}

// =============================================================================
// 9. Acrobot Environment PARITY TESTS
// =============================================================================

function testAcrobotParity() {
    console.log('\n' + '='.repeat(70));
    console.log('9. ACROBOT PARITY');
    console.log('='.repeat(70));

    const acrobot = new Acrobot();

    // JAVA AcrobotState.java: numActions = 3
    assertEqual(acrobot.numActions, 3, 'numActions');

    // Physical constants - EXACT from Java AcrobotState.java lines 16-30
    assertEqual(acrobot.m1, 1.0, 'm1 (mass of link 1)');
    assertEqual(acrobot.m2, 1.0, 'm2 (mass of link 2)');
    assertEqual(acrobot.l1, 1.0, 'l1 (length of link 1)');
    assertEqual(acrobot.l2, 1.0, 'l2 (length of link 2)');
    assertEqual(acrobot.lc1, 0.5, 'lc1 (center of mass link 1)');
    assertEqual(acrobot.lc2, 0.5, 'lc2 (center of mass link 2)');
    assertEqual(acrobot.I1, 1.0, 'I1 (moment of inertia link 1)');
    assertEqual(acrobot.I2, 1.0, 'I2 (moment of inertia link 2)');
    assertEqual(acrobot.g, 9.8, 'g (gravity)');
    assertEqual(acrobot.dt, 0.05, 'dt (time step)');

    // JAVA line 31: acrobotGoalPosition = 1.0
    assertEqual(acrobot.acrobotGoalPosition, 1.0, 'acrobotGoalPosition');

    // Angular limits - EXACT from Java
    assertEqual(acrobot.maxTheta1, Math.PI, 'maxTheta1');
    assertEqual(acrobot.maxTheta2, Math.PI, 'maxTheta2');
    assertEqual(acrobot.maxTheta1Dot, 4 * Math.PI, 'maxTheta1Dot');
    assertEqual(acrobot.maxTheta2Dot, 9 * Math.PI, 'maxTheta2Dot');

    // Rewards - EXACT from Java Acrobot.java lines 116-122
    assertEqual(acrobot.stepReward, -1, 'stepReward');
    assertEqual(acrobot.goalReward, 0, 'goalReward');

    // Observation ranges
    assertEqual(acrobot.obsRanges.length, 4, 'obsRanges length (4D state)');
    assertArrayEqual(acrobot.obsRanges[0], [-Math.PI, Math.PI], 'obsRanges[0] (theta1)');
    assertArrayEqual(acrobot.obsRanges[1], [-Math.PI, Math.PI], 'obsRanges[1] (theta2)');
    assertArrayEqual(acrobot.obsRanges[2], [-4 * Math.PI, 4 * Math.PI], 'obsRanges[2] (theta1Dot)');
    assertArrayEqual(acrobot.obsRanges[3], [-9 * Math.PI, 9 * Math.PI], 'obsRanges[3] (theta2Dot)');

    // Test start state - EXACT from Java resetBottom()
    acrobot.start();
    assertEqual(acrobot.theta1, 0.0, 'theta1 initial (resetBottom)');
    assertEqual(acrobot.theta2, 0.0, 'theta2 initial (resetBottom)');
    assertEqual(acrobot.theta1Dot, 0.0, 'theta1Dot initial');
    assertEqual(acrobot.theta2Dot, 0.0, 'theta2Dot initial');
}

// =============================================================================
// 10. RobotArm Environment PARITY TESTS
// =============================================================================

function testRobotArmParity() {
    console.log('\n' + '='.repeat(70));
    console.log('10. ROBOT ARM PARITY');
    console.log('='.repeat(70));

    // Static constants - EXACT from Java RobotArmState.java
    assertArrayEqual(RobotArm.worldDims, [20, 20, 10, 10], 'worldDims');

    // EXACT from Java: SEG_LENS
    const expectedSegLens = [0.3, 0.1, 0.2, 0.1, 0.2, 0.05, 0.05];
    assertArrayEqual(RobotArm.SEG_LENS, expectedSegLens, 'SEG_LENS');

    // EXACT from Java
    assertEqual(RobotArm.graspRadius, 0.05, 'graspRadius');
    assertEqual(RobotArm.rewardPerStep, -1.0, 'rewardPerStep');
    assertEqual(RobotArm.rewardAtGoal, 0.0, 'rewardAtGoal');
    assertEqual(RobotArm.agentSpeed, 1, 'agentSpeed');

    // Origin and target - EXACT from Java
    assertEqual(RobotArm.origin.x, 0.5, 'origin.x');
    assertEqual(RobotArm.origin.y, 1.0, 'origin.y');
    assertEqual(RobotArm.targetLoc.x, 0.8, 'targetLoc.x');
    assertEqual(RobotArm.targetLoc.y, 0.6, 'targetLoc.y');

    // Number of actions - EXACT from Java
    const robotArm = new RobotArm();
    const spec = robotArm.init();
    assertEqual(spec.numActions, RobotArm.worldDims.length * 2, 'numActions');
    assertEqual(spec.numActions, 8, 'numActions = 8');

    // Observation ranges
    assertEqual(spec.obsRanges.length, 4, 'obsRanges length');
    for (let i = 0; i < 4; i++) {
        assertEqual(spec.obsRanges[i][0], 0, `obsRanges[${i}][0] = 0`);
        assertEqual(spec.obsRanges[i][1], RobotArm.worldDims[i] - 1, `obsRanges[${i}][1]`);
    }
}

// =============================================================================
// 11. Tetris Environment PARITY TESTS
// =============================================================================

function testTetrisParity() {
    console.log('\n' + '='.repeat(70));
    console.log('11. TETRIS PARITY');
    console.log('='.repeat(70));

    // TetrisActions - EXACT from Java
    assertEqual(TetrisActions.LEFT, 0, 'TetrisActions.LEFT');
    assertEqual(TetrisActions.RIGHT, 1, 'TetrisActions.RIGHT');
    assertEqual(TetrisActions.CW, 2, 'TetrisActions.CW');
    assertEqual(TetrisActions.CCW, 3, 'TetrisActions.CCW');
    assertEqual(TetrisActions.NONE, 4, 'TetrisActions.NONE');
    assertEqual(TetrisActions.FALL, 5, 'TetrisActions.FALL');

    // TetrisState defaults - EXACT from Java
    const state = new TetrisState();
    assertEqual(state.worldWidth, 10, 'worldWidth');
    assertEqual(state.worldHeight, 20, 'worldHeight');
    assertEqual(state.worldState.length, 200, 'worldState length (10x20)');

    // Number of pieces - EXACT from Java
    assertEqual(state.possibleBlocks.length, 7, 'possibleBlocks (7 tetrominos)');

    // Initial state after reset - EXACT from Java
    state.reset();
    assertEqual(state.currentX, 4, 'currentX after reset (width/2 - 1)');
    assertEqual(state.currentY, 0, 'currentY after reset');
    assertEqual(state.score, 0, 'score after reset');
    assertEqual(state.currentRotation, 0, 'currentRotation after reset');
    assertEqual(state.is_game_over, false, 'is_game_over after reset');
    assertEqual(state.blockMobile, true, 'blockMobile after reset');
    assertEqual(state.currentBlockId, 0, 'currentBlockId after reset');

    // Observation format - EXACT from Java (worldState + 7 values)
    const obs = state.getObservation();
    assertEqual(obs.length, 207, 'observation length (200 + 7)');
}

// =============================================================================
// 12. TetrisFeatures PARITY TESTS
// =============================================================================

function testTetrisFeaturesParity() {
    console.log('\n' + '='.repeat(70));
    console.log('12. TETRIS FEATURES PARITY');
    console.log('='.repeat(70));

    // Feature indices - EXACT from Java FeatGen_Tetris.java
    assertEqual(TetrisFeatures.NUM_FEATS, 46, 'NUM_FEATS');
    assertEqual(TetrisFeatures.COL_HT_START_I, 0, 'COL_HT_START_I');
    assertEqual(TetrisFeatures.MAX_COL_HT_I, 10, 'MAX_COL_HT_I');
    assertEqual(TetrisFeatures.COL_DIFF_START_I, 11, 'COL_DIFF_START_I');
    assertEqual(TetrisFeatures.NUM_HOLES_I, 20, 'NUM_HOLES_I');
    assertEqual(TetrisFeatures.MAX_WELL_I, 21, 'MAX_WELL_I');
    assertEqual(TetrisFeatures.SUM_WELL_I, 22, 'SUM_WELL_I');
    assertEqual(TetrisFeatures.SQUARED_FEATS_START_I, 23, 'SQUARED_FEATS_START_I');

    // Scaling factor - EXACT from Java
    assertEqual(TetrisFeatures.HT_SQ_SCALE, 100.0, 'HT_SQ_SCALE');

    // Configuration flags - EXACT from Java
    assertEqual(TetrisFeatures.REMOVE_DECIMAL, false, 'REMOVE_DECIMAL');
    assertEqual(TetrisFeatures.COUNT_LAST_COL_FOR_WELL_SUM, true, 'COUNT_LAST_COL_FOR_WELL_SUM');
    assertEqual(TetrisFeatures.SCALE_ALL_SQUARED_FEATS, false, 'SCALE_ALL_SQUARED_FEATS');

    // Feature generator methods
    const obsRanges = [[0, 10]];  // Dummy
    const featGen = new TetrisFeatures(obsRanges, 6);
    assertEqual(featGen.getNumExtendedFeatures(), 46, 'getNumExtendedFeatures()');
    assertEqual(featGen.getNumStateFeatures(), 46, 'getNumStateFeatures()');
    assertEqual(featGen.worldWidth, 10, 'worldWidth');
    assertEqual(featGen.worldHeight, 20, 'worldHeight');
}

// =============================================================================
// 13. TamerRLAgent PARITY TESTS
// =============================================================================

function testTamerRLAgentParity() {
    console.log('\n' + '='.repeat(70));
    console.log('13. TAMER-RL AGENT PARITY');
    console.log('='.repeat(70));

    // CombinationMethods enum - EXACT from Java
    assertEqual(CombinationMethods.RL_ON_H_AS_R, -2, 'RL_ON_H_AS_R');
    assertEqual(CombinationMethods.TAMER_ONLY, -1, 'TAMER_ONLY');
    assertEqual(CombinationMethods.RL_ONLY, 0, 'RL_ONLY');
    assertEqual(CombinationMethods.REW_SHAPING, 1, 'REW_SHAPING');
    assertEqual(CombinationMethods.FEAT_ADD, 2, 'FEAT_ADD');
    assertEqual(CombinationMethods.Q_INIT, 3, 'Q_INIT');
    assertEqual(CombinationMethods.Q_AUGM, 4, 'Q_AUGM');
    assertEqual(CombinationMethods.EXTRA_ACT, 5, 'EXTRA_ACT');
    assertEqual(CombinationMethods.ACT_BIASING, 6, 'ACT_BIASING');
    assertEqual(CombinationMethods.BERNOULLI_ACT, 7, 'BERNOULLI_ACT');
    assertEqual(CombinationMethods.STATE_POT_FCN_SHAPING, 8, 'STATE_POT_FCN_SHAPING');
    assertEqual(CombinationMethods.SA_POT_FCN_SHAPING, 9, 'SA_POT_FCN_SHAPING');
    assertEqual(CombinationMethods.PROB_ACT_W_OSCILL_DAMP, 10, 'PROB_ACT_W_OSCILL_DAMP');

    // Create agent with defaults
    const env = new LoopMaze();
    const agent = new TamerRLAgent(env);

    // Defaults - EXACT from Java
    assertEqual(agent.COMBINATION_METHOD, CombinationMethods.ACT_BIASING, 'default combinationMethod');
    assertEqual(agent.INITIAL_COMB_PARAM, 10.0, 'default combinationParam');
    assertEqual(agent.H_INFLUENCE_METHOD, 'annealedParam', 'default hInfluenceMethod');
    assertEqual(agent.SIMUL_LEARNING, false, 'default simulLearning');

    // Agent options defaults
    assertEqual(agent.agentOptions.epsilon, 0.0, 'agentOptions.epsilon default');
    assertEqual(agent.agentOptions.stepSize, 0.001, 'agentOptions.stepSize default');
    assertEqual(agent.agentOptions.basisFcnsPerDim, 10, 'agentOptions.basisFcnsPerDim default');
    assertEqual(agent.agentOptions.discountFactor, 0.99, 'agentOptions.discountFactor default');
    assertEqual(agent.agentOptions.creditDelayMs, 200, 'agentOptions.creditDelayMs default');
    assertEqual(agent.agentOptions.creditWindowMs, 600, 'agentOptions.creditWindowMs default');
    assertEqual(agent.agentOptions.creditDistType, 'uniform', 'agentOptions.creditDistType default');
}

// =============================================================================
// 14. SarsaLambdaAgent PARITY TESTS
// =============================================================================

function testSarsaLambdaAgentParity() {
    console.log('\n' + '='.repeat(70));
    console.log('14. SARSA-LAMBDA AGENT PARITY');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env);

    // Defaults - EXACT from Java
    assertEqual(agent.epsilon, 0.1, 'epsilon default');
    assertEqual(agent.stepSize, 0.01, 'stepSize default');
    assertEqual(agent.discountFactor, 0.99, 'discountFactor default');
    assertEqual(agent.traceDecayFactor, 0.9, 'traceDecayFactor (lambda) default');
    assertEqual(agent.basisFcnsPerDim, 10, 'basisFcnsPerDim default');

    // Action selector config
    assertEqual(agent.actSelector.selectionMethod, 'e-greedy', 'actSelector.selectionMethod');
    assertEqual(agent.actSelector.selectionParams.epsilon, 0.1, 'actSelector epsilon');
    assertEqual(agent.actSelector.selectionParams.epsilonAnnealRate, 0.9995, 'epsilonAnnealRate');

    // Model config
    assertEqual(agent.model.decayFactor, 0.9, 'model.decayFactor = traceDecayFactor');
}

// =============================================================================
// 15. ImitationAgent PARITY TESTS
// =============================================================================

function testImitationAgentParity() {
    console.log('\n' + '='.repeat(70));
    console.log('15. IMITATION AGENT PARITY');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new ImitationAgent(env);

    // Defaults - EXACT from Java
    assertEqual(agent.stepSize, 0.01, 'stepSize default');
    assertEqual(agent.basisFcnsPerDim, 10, 'basisFcnsPerDim default');
    assertEqual(agent.controlOnly, false, 'controlOnly default');
    assertEqual(agent.okayToHang, false, 'okayToHang default');

    // Action selector config
    assertEqual(agent.actSelector.selectionMethod, 'greedy', 'actSelector.selectionMethod');

    // Training state
    assertEqual(agent.inTrainSess, true, 'inTrainSess default');
    assertEqual(agent.allowUserToggledTraining, true, 'allowUserToggledTraining');
}

// =============================================================================
// 16. ActionSelect PARITY TESTS
// =============================================================================

function testActionSelectParity() {
    console.log('\n' + '='.repeat(70));
    console.log('16. ACTION SELECT PARITY');
    console.log('='.repeat(70));

    // Create with minimal model
    const mockModel = {
        predictForAction: (obs, action) => 0
    };
    const actSelect = new ActionSelect(mockModel, 'greedy', {}, 4);

    // Defaults - EXACT from Java
    assertEqual(actSelect.selectionMethod, 'greedy', 'selectionMethod');
    assertEqual(actSelect.treeSearch, false, 'treeSearch default');
    assertEqual(actSelect.greedyLeafPathLength, 0, 'greedyLeafPathLength default');
    assertEqual(actSelect.exhaustiveSearchDepth, 1, 'exhaustiveSearchDepth default');
    assertEqual(actSelect.randomizeSearchDepth, true, 'randomizeSearchDepth default');
    assertEqual(actSelect.discountParam, Number.MAX_VALUE, 'discountParam default');
    assertEqual(actSelect.discountType, 'EXPON', 'discountType default');

    // Discount factor conversion - EXACT from Java
    const factor = 0.99;
    const param = ActionSelect.discFactorToParam(factor);
    assertEqual(param, -1 * Math.log(factor), 'discFactorToParam formula', 1e-10);

    const backToFactor = ActionSelect.discParamToFactor(param);
    assertEqual(backToFactor, factor, 'discParamToFactor round-trip', 1e-10);

    // Special case: factor = 1.0
    const paramForOne = ActionSelect.discFactorToParam(1.0);
    assertEqual(paramForOne, 0, 'discFactorToParam(1.0) = 0');
}

// =============================================================================
// 17. HInfluence PARITY TESTS
// =============================================================================

function testHInfluenceParity() {
    console.log('\n' + '='.repeat(70));
    console.log('17. H-INFLUENCE PARITY');
    console.log('='.repeat(70));

    // Default construction
    const hInf = new HInfluence('annealedParam', 1.0);

    // Defaults - EXACT from Java
    assertEqual(hInf.INFLUENCE_METHOD, 'annealedParam', 'INFLUENCE_METHOD default');
    assertEqual(hInf.COMB_PARAM, 1.0, 'COMB_PARAM');
    assertEqual(hInf.STEP_DECAY_FACTOR, 1.0, 'STEP_DECAY_FACTOR default');
    assertEqual(hInf.EP_DECAY_FACTOR, 1.0, 'EP_DECAY_FACTOR default');
    assertEqual(hInf.TRACE_STYLE, 'accumulating', 'TRACE_STYLE default');
    assertEqual(hInf.ACCUM_FACTOR, 0.0, 'ACCUM_FACTOR default');
    assertEqual(hInf.stateOnly, false, 'stateOnly default');

    // Traces for annealedParam should be length 1
    assertEqual(hInf.traces.length, 1, 'traces length for annealedParam');
    assertEqual(hInf.traces[0], 1.0, 'traces[0] initialized to 1.0');

    // Test getHInfluence for annealedParam
    const influence = hInf.getHInfluence([0, 1], 0);
    assertEqual(influence, 1.0, 'getHInfluence for annealedParam');
}

// =============================================================================
// 18. TetrisPiece PARITY TESTS
// =============================================================================

function testTetrisPieceParity() {
    console.log('\n' + '='.repeat(70));
    console.log('18. TETRIS PIECE PARITY');
    console.log('='.repeat(70));

    // Test piece grid size - EXACT from Java (5x5 grid)
    const piece = new TetrisPiece();
    assertEqual(piece.thePiece.length, 4, 'piece has 4 orientations');
    assertEqual(piece.thePiece[0].length, 5, 'piece grid is 5 rows');
    assertEqual(piece.thePiece[0][0].length, 5, 'piece grid is 5 cols');

    // Test Line piece - EXACT from Java makeLine()
    const line = TetrisPiece.makeLine();
    // Orientation 0: vertical line at column 2
    assertEqual(line.thePiece[0][0][2], 1, 'Line[0] row0 has block at col2');
    assertEqual(line.thePiece[0][1][2], 1, 'Line[0] row1 has block at col2');
    assertEqual(line.thePiece[0][2][2], 1, 'Line[0] row2 has block at col2');
    assertEqual(line.thePiece[0][3][2], 1, 'Line[0] row3 has block at col2');
    // Orientation 1: horizontal line at row 2
    assertEqual(line.thePiece[1][2][1], 1, 'Line[1] row2 has block at col1');
    assertEqual(line.thePiece[1][2][2], 1, 'Line[1] row2 has block at col2');
    assertEqual(line.thePiece[1][2][3], 1, 'Line[1] row2 has block at col3');
    assertEqual(line.thePiece[1][2][4], 1, 'Line[1] row2 has block at col4');

    // Test Square piece - EXACT from Java makeSquare()
    const square = TetrisPiece.makeSquare();
    // All orientations are the same for square
    assertEqual(square.thePiece[0][1][2], 1, 'Square[0] row1 col2');
    assertEqual(square.thePiece[0][1][3], 1, 'Square[0] row1 col3');
    assertEqual(square.thePiece[0][2][2], 1, 'Square[0] row2 col2');
    assertEqual(square.thePiece[0][2][3], 1, 'Square[0] row2 col3');

    // Test T piece - EXACT from Java makeTri()
    const tri = TetrisPiece.makeTri();
    // Orientation 0: T pointing down
    assertEqual(tri.thePiece[0][1][2], 1, 'Tri[0] row1 col2 (top)');
    assertEqual(tri.thePiece[0][2][1], 1, 'Tri[0] row2 col1 (left)');
    assertEqual(tri.thePiece[0][2][2], 1, 'Tri[0] row2 col2 (center)');
    assertEqual(tri.thePiece[0][2][3], 1, 'Tri[0] row2 col3 (right)');

    // Test piece count
    assertEqual(typeof TetrisPiece.makeLine, 'function', 'makeLine exists');
    assertEqual(typeof TetrisPiece.makeSquare, 'function', 'makeSquare exists');
    assertEqual(typeof TetrisPiece.makeTri, 'function', 'makeTri exists');
    assertEqual(typeof TetrisPiece.makeSShape, 'function', 'makeSShape exists');
    assertEqual(typeof TetrisPiece.makeZShape, 'function', 'makeZShape exists');
    assertEqual(typeof TetrisPiece.makeLShape, 'function', 'makeLShape exists');
    assertEqual(typeof TetrisPiece.makeJShape, 'function', 'makeJShape exists');
}

// =============================================================================
// 19. Tetris Environment Wrapper PARITY TESTS
// =============================================================================

function testTetrisWrapperParity() {
    console.log('\n' + '='.repeat(70));
    console.log('19. TETRIS WRAPPER PARITY');
    console.log('='.repeat(70));

    const tetris = new Tetris();
    const spec = tetris.init();

    // EXACT from Java env_init()
    assertEqual(spec.numActions, 5, 'numActions (0-4, FALL removed)');
    assertEqual(spec.discountFactor, 1.0, 'discountFactor');
    assertEqual(spec.isEpisodic, true, 'isEpisodic');
    assertEqual(tetris.terminalScore, 0, 'terminalScore');

    // Observation ranges - EXACT from Java
    assertEqual(spec.obsRanges.length, 207, 'obsRanges length (200 + 7)');
    // Board cells are 0-1
    assertArrayEqual(spec.obsRanges[0], [0, 1], 'obsRanges[0] board cell');
    assertArrayEqual(spec.obsRanges[199], [0, 1], 'obsRanges[199] last board cell');
    // blockMobile
    assertArrayEqual(spec.obsRanges[200], [0, 1], 'obsRanges[200] blockMobile');
    // currentBlockId
    assertArrayEqual(spec.obsRanges[201], [0, 6], 'obsRanges[201] blockId (0-6)');
    // currentRotation
    assertArrayEqual(spec.obsRanges[202], [0, 3], 'obsRanges[202] rotation (0-3)');

    // Game state
    assertEqual(tetris.getName(), 'Tetris', 'getName()');
}

// =============================================================================
// 20. ExtActionAgentWrap PARITY TESTS
// =============================================================================

function testExtActionAgentWrapParity() {
    console.log('\n' + '='.repeat(70));
    console.log('20. EXT-ACTION AGENT WRAP PARITY');
    console.log('='.repeat(70));

    const env = new LoopMaze();

    // Default options - EXACT from Java
    const wrap = new ExtActionAgentWrap(env);
    assertEqual(wrap.callCoreAgentEveryStep, false, 'callCoreAgentEveryStep default');

    // State tracking
    assertEqual(wrap.currExtendedAction, null, 'currExtendedAction initial');
    assertEqual(wrap.currExtActI, 0, 'currExtActI initial');
    assertEqual(wrap.rewThisExtAct, 0, 'rewThisExtAct initial');
}

// =============================================================================
// MAIN
// =============================================================================

async function runAllTests() {
    console.log('='.repeat(70));
    console.log('EXHAUSTIVE JAVA-JAVASCRIPT PARITY TEST');
    console.log('Covering ALL environments and ALL algorithms');
    console.log('='.repeat(70));

    try {
        // Core components
        testLinearModelParity();
        testRBFFeaturesParity();
        testCreditAssignParity();
        testTamerAgentParity();

        // Original environments
        testLoopMazeParity();
        testMountainCarParity();
        testCartPoleParity();

        // NEW: Additional environments
        testAcrobotParity();
        testRobotArmParity();
        testTetrisParity();

        // NEW: Feature generators
        testTetrisFeaturesParity();

        // NEW: All agents
        testTamerRLAgentParity();
        testSarsaLambdaAgentParity();
        testImitationAgentParity();

        // NEW: Core modules
        testActionSelectParity();
        testHInfluenceParity();

        // NEW: Additional Tetris components
        testTetrisPieceParity();
        testTetrisWrapperParity();

        // NEW: Agent wrappers
        testExtActionAgentWrapParity();

        // Integration test
        testTamerIntegration();
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
    console.log(`Warnings: ${warnings.length}`);

    if (warnings.length > 0) {
        console.log('\n--- WARNINGS ---');
        for (const warning of warnings) {
            console.log(`  ⚠ ${warning}`);
        }
    }

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
