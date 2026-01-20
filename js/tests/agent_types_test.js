/**
 * Agent Types Configuration Test
 *
 * Verifies configuration for ALL agent types:
 * - TamerAgent
 * - SarsaLambdaAgent
 * - TamerRLAgent
 * - ImitationAgent
 * - ExtActionAgentWrap
 *
 * Run with: node --experimental-vm-modules js/tests/agent_types_test.js
 */

import { TamerAgent } from '../core/TamerAgent.js';
import { SarsaLambdaAgent } from '../agents/SarsaLambdaAgent.js';
import { TamerRLAgent, CombinationMethods } from '../agents/TamerRLAgent.js';
import { ImitationAgent } from '../agents/ImitationAgent.js';
import { ExtActionAgentWrap } from '../agents/ExtActionAgentWrap.js';

import { RBFFeatures } from '../features/RBFFeatures.js';
import { TetrisFeatures } from '../features/TetrisFeatures.js';

import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';
import { CartPole } from '../environments/CartPole.js';
import { Tetris } from '../environments/tetris/index.js';

// =============================================================================
// EXPECTED CONFIGURATION FROM JAVA
// =============================================================================

const JAVA_SARSA_CONFIG = {
    // From Params.java SarsaLambdaAgent section
    mountaincar: {
        basisFcnsPerDim: 34,
        relWidth: 0.062,
        stepSize: 0.05,
        traceDecayFactor: 0.9,
        epsilon: 0.0
    },
    cartpole: {
        basisFcnsPerDim: 8,
        relWidth: 0.13,
        stepSize: 0.05,
        traceDecayFactor: 0.86,
        epsilon: 0.085,
        epsilonAnnealRate: 0.9995
    }
};

const JAVA_HINFLUENCE_CONFIG = {
    // From Params.java HInfluence section
    cartpole: {
        basisFcnsPerDim: 8,
        relWidth: 0.08,
        stepDecayFactor: 0.99996,  // with elig traces
        epDecayFactor: 0.98,
        accumFactor: 0.2
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

// =============================================================================
// TAMER AGENT TESTS
// =============================================================================

function testTamerAgent() {
    console.log('\n' + '='.repeat(70));
    console.log('TAMER AGENT TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const env = new LoopMaze();
    const agent = new TamerAgent(env, {});

    assertEqual(agent.epsilon, 0.0, 'Default epsilon');
    assertEqual(agent.basisFcnsPerDim, 40, 'Default basisFcnsPerDim');
    assertEqual(agent.featGen.relWidth, 0.08, 'Default relWidth');

    // Verify model configuration
    assertEqual(agent.model.useBiasWt, true, 'Model has bias weight');
    assertEqual(agent.model.decayFactor, 0.0, 'No eligibility traces');

    // Verify feature generator
    assertEqual(agent.featGen instanceof RBFFeatures, true, 'Uses RBF features');
    assertEqual(agent.featGen.normBounds[0], -1, 'normBounds min = -1');
    assertEqual(agent.featGen.normBounds[1], 1, 'normBounds max = 1');
    assertEqual(agent.featGen.addBiasFeatPerAct, true, 'Has bias per action');
    assertEqual(agent.featGen.biasFeatVal, 0.1, 'Bias feature value = 0.1');

    console.log('\n--- Credit Assignment ---');
    assertEqual(agent.creditDistType, 'uniform', 'Default credit dist type');
    assertEqual(agent.creditDelayMs, 200, 'Default credit delay');
    assertEqual(agent.creditWindowMs, 600, 'Default credit window');
}

// =============================================================================
// SARSA LAMBDA AGENT TESTS
// =============================================================================

function testSarsaLambdaAgent() {
    console.log('\n' + '='.repeat(70));
    console.log('SARSA LAMBDA AGENT TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const env = new LoopMaze();
    const agent = new SarsaLambdaAgent(env, {});

    assertEqual(agent.epsilon, 0.1, 'Default epsilon');
    assertEqual(agent.stepSize, 0.01, 'Default stepSize');
    assertEqual(agent.discountFactor, 0.99, 'Default discount factor');
    assertEqual(agent.traceDecayFactor, 0.9, 'Default trace decay (lambda)');
    assertEqual(agent.basisFcnsPerDim, 10, 'Default basisFcnsPerDim');

    // Verify feature generator
    assertEqual(agent.featGen instanceof RBFFeatures, true, 'Uses RBF features');
    assertEqual(agent.featGen.relWidth, 0.08, 'Default relWidth');
    assertEqual(agent.featGen.normBounds[0], -1, 'normBounds min = -1');
    assertEqual(agent.featGen.normBounds[1], 1, 'normBounds max = 1');

    // Verify model has eligibility traces enabled
    assertEqual(agent.model.decayFactor, 0.9, 'Model has trace decay');

    console.log('\n--- CartPole Configuration (Java Params) ---');
    const cartpoleEnv = new CartPole();
    const cartpoleSarsa = new SarsaLambdaAgent(cartpoleEnv, {
        basisFcnsPerDim: JAVA_SARSA_CONFIG.cartpole.basisFcnsPerDim,
        stepSize: JAVA_SARSA_CONFIG.cartpole.stepSize,
        traceDecayFactor: JAVA_SARSA_CONFIG.cartpole.traceDecayFactor,
        epsilon: JAVA_SARSA_CONFIG.cartpole.epsilon
    });

    // Create custom RBF with Java relWidth
    assertEqual(cartpoleSarsa.basisFcnsPerDim, 8, 'CartPole basisFcnsPerDim');
    assertEqual(cartpoleSarsa.stepSize, 0.05, 'CartPole stepSize');
    assertEqual(cartpoleSarsa.traceDecayFactor, 0.86, 'CartPole trace decay');
    assertEqual(cartpoleSarsa.epsilon, 0.085, 'CartPole epsilon');
}

// =============================================================================
// TAMER-RL AGENT TESTS
// =============================================================================

function testTamerRLAgent() {
    console.log('\n' + '='.repeat(70));
    console.log('TAMER-RL AGENT TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const env = new LoopMaze();
    const agent = new TamerRLAgent(env, {});

    // Verify combination methods enum
    assertEqual(CombinationMethods.RL_ON_H_AS_R, -2, 'RL_ON_H_AS_R = -2');
    assertEqual(CombinationMethods.TAMER_ONLY, -1, 'TAMER_ONLY = -1');
    assertEqual(CombinationMethods.RL_ONLY, 0, 'RL_ONLY = 0');
    assertEqual(CombinationMethods.REW_SHAPING, 1, 'REW_SHAPING = 1');
    assertEqual(CombinationMethods.FEAT_ADD, 2, 'FEAT_ADD = 2');
    assertEqual(CombinationMethods.Q_AUGM, 4, 'Q_AUGM = 4');
    assertEqual(CombinationMethods.ACT_BIASING, 6, 'ACT_BIASING = 6');
    assertEqual(CombinationMethods.BERNOULLI_ACT, 7, 'BERNOULLI_ACT = 7');

    // Default combination method should be ACT_BIASING (6)
    // Note: Property name is COMBINATION_METHOD (uppercase)
    assertEqual(agent.COMBINATION_METHOD, CombinationMethods.ACT_BIASING,
        'Default combination method is ACT_BIASING');

    // Verify it has both TAMER and RL components
    assertEqual(agent.tamerAgent !== undefined, true, 'Has TAMER agent');
    assertEqual(agent.rlAgent !== undefined, true, 'Has RL agent');
}

// =============================================================================
// IMITATION AGENT TESTS
// =============================================================================

function testImitationAgent() {
    console.log('\n' + '='.repeat(70));
    console.log('IMITATION AGENT TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Default Configuration ---');
    const env = new LoopMaze();
    const agent = new ImitationAgent(env, {});

    // Verify it uses RBF features like TAMER
    assertEqual(agent.featGen instanceof RBFFeatures, true, 'Uses RBF features');

    // Verify default parameters
    // Note: ImitationAgent defaults to 10, different from TamerAgent's 40
    assertEqual(agent.basisFcnsPerDim, 10, 'Default basisFcnsPerDim (10 for ImitationAgent)');
    assertEqual(agent.featGen.relWidth, 0.08, 'Default relWidth');
}

// =============================================================================
// EXT ACTION AGENT WRAP TESTS
// =============================================================================

function testExtActionAgentWrap() {
    console.log('\n' + '='.repeat(70));
    console.log('EXT ACTION AGENT WRAP TESTS');
    console.log('='.repeat(70));

    console.log('\n--- Tetris Configuration ---');
    const env = new Tetris();
    const agent = new ExtActionAgentWrap(env, {
        coreAgentClass: TamerAgent,
        coreAgentOptions: {
            FeatGenClass: TetrisFeatures,
            epsilon: 0.0,
            stepSize: 0.000005 / 47,
            creditDistType: 'previousStep',
            creditDelayMs: 200,
            creditWindowMs: 600
        }
    });
    agent.init();

    // Verify core agent is created
    assertEqual(agent.coreAgent !== null, true, 'Core agent created');
    assertEqual(agent.coreAgent instanceof TamerAgent, true, 'Core agent is TamerAgent');

    // Verify Tetris-specific config
    assertEqual(agent.coreAgent.creditDistType, 'previousStep',
        'Credit dist type is previousStep');
    assertEqual(agent.coreAgent.featGen instanceof TetrisFeatures, true,
        'Uses TetrisFeatures');
    assertEqual(agent.coreAgent.featGen.getNumExtendedFeatures(), 46,
        'Tetris has 46 features');

    // Verify step size
    const expectedStepSize = 0.000005 / 47;
    assertEqual(Math.abs(agent.coreAgent.stepSize - expectedStepSize) < 1e-15, true,
        'Step size matches Java (0.000005/47)');
}

// =============================================================================
// MAIN
// =============================================================================

async function runAllTests() {
    console.log('='.repeat(70));
    console.log('AGENT TYPES CONFIGURATION TEST');
    console.log('Verifying all agent types match Java configuration');
    console.log('='.repeat(70));

    try {
        testTamerAgent();
        testSarsaLambdaAgent();
        testTamerRLAgent();
        testImitationAgent();
        testExtActionAgentWrap();
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
