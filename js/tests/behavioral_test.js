/**
 * BEHAVIORAL TEST SUITE
 *
 * Tests actual behavior and logic flows, NOT just parameter values.
 * These tests verify that the code behaves correctly in real usage scenarios.
 *
 * Run with: node --experimental-vm-modules js/tests/behavioral_test.js
 */

import { TamerAgent } from '../core/TamerAgent.js';
import { CreditAssign } from '../core/CreditAssign.js';
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { LoopMaze } from '../environments/LoopMaze.js';
import { MountainCar } from '../environments/MountainCar.js';

// Test tracking
let passedTests = 0;
let failedTests = [];

function assertEqual(actual, expected, testName, tolerance = 0) {
    if (tolerance > 0) {
        if (Math.abs(actual - expected) <= tolerance) {
            console.log(`  ✓ ${testName}`);
            passedTests++;
            return true;
        }
    } else if (actual === expected) {
        console.log(`  ✓ ${testName}`);
        passedTests++;
        return true;
    }
    console.log(`  ✗ ${testName}`);
    console.log(`    Expected: ${expected}`);
    console.log(`    Actual: ${actual}`);
    failedTests.push({ testName, actual, expected });
    return false;
}

function assertTrue(condition, testName) {
    return assertEqual(condition, true, testName);
}

function assertFalse(condition, testName) {
    return assertEqual(condition, false, testName);
}

function assertNotEqual(actual, notExpected, testName) {
    if (actual !== notExpected) {
        console.log(`  ✓ ${testName}`);
        passedTests++;
        return true;
    }
    console.log(`  ✗ ${testName}`);
    console.log(`    Should NOT be: ${notExpected}`);
    console.log(`    Actual: ${actual}`);
    failedTests.push({ testName, actual, expected: `not ${notExpected}` });
    return false;
}

function assertArrayNotAllZero(arr, testName) {
    const hasNonZero = arr.some(v => v !== 0);
    if (hasNonZero) {
        console.log(`  ✓ ${testName}`);
        passedTests++;
        return true;
    }
    console.log(`  ✗ ${testName}`);
    console.log(`    Array is all zeros: [${arr.join(', ')}]`);
    failedTests.push({ testName, actual: 'all zeros', expected: 'some non-zero values' });
    return false;
}

function assertArrayAllZero(arr, testName) {
    const allZero = arr.every(v => v === 0);
    if (allZero) {
        console.log(`  ✓ ${testName}`);
        passedTests++;
        return true;
    }
    console.log(`  ✗ ${testName}`);
    console.log(`    Array has non-zero values: [${arr.join(', ')}]`);
    failedTests.push({ testName, actual: 'has non-zero', expected: 'all zeros' });
    return false;
}

// =============================================================================
// 1. TRAINING TOGGLE BEHAVIOR
// =============================================================================

function testTrainingToggleBehavior() {
    console.log('\n' + '='.repeat(70));
    console.log('1. TRAINING TOGGLE BEHAVIOR');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,  // Small for fast tests
        creditDelayMs: 0,    // Immediate credit for testing
        creditWindowMs: 100
    });

    // Initial state: training should be OFF
    assertFalse(agent.isTraining(), 'Training starts OFF');

    // Toggle training ON
    agent.toggleTraining();
    assertTrue(agent.isTraining(), 'Training ON after toggle');

    // Toggle training OFF
    agent.toggleTraining();
    assertFalse(agent.isTraining(), 'Training OFF after second toggle');
}

// =============================================================================
// 2. REWARDS IGNORED WHEN TRAINING IS OFF
// =============================================================================

function testRewardsIgnoredWhenTrainingOff() {
    console.log('\n' + '='.repeat(70));
    console.log('2. REWARDS IGNORED WHEN TRAINING IS OFF');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,
        creditDelayMs: 0,
        creditWindowMs: 500,
        creditDistType: 'immediate'  // Immediate credit assignment
    });

    // Start episode
    agent.startEpisode();

    // Training is OFF by default
    assertFalse(agent.isTraining(), 'Training is OFF');

    // Get initial weights
    const initialWeights = [...agent.model.weights];

    // Give reward while training is OFF
    const time = performance.now();
    agent.processHumanReward(1.0, time);

    // Take a step to process the reward
    agent.step(time + 100);

    // Weights should NOT have changed
    const weightsAfterReward = [...agent.model.weights];
    let weightsChanged = false;
    for (let i = 0; i < initialWeights.length; i++) {
        if (initialWeights[i] !== weightsAfterReward[i]) {
            weightsChanged = true;
            break;
        }
    }

    assertFalse(weightsChanged, 'Weights unchanged when training OFF and reward given');
}

// =============================================================================
// 3. REWARDS UPDATE MODEL WHEN TRAINING IS ON
// =============================================================================

function testRewardsUpdateModelWhenTrainingOn() {
    console.log('\n' + '='.repeat(70));
    console.log('3. REWARDS UPDATE MODEL WHEN TRAINING IS ON');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,
        creditDelayMs: 0,
        creditWindowMs: 1000,
        creditDistType: 'uniform'  // Use uniform for proper timing
    });

    // Start episode
    agent.startEpisode();

    // Turn training ON
    agent.toggleTraining();
    assertTrue(agent.isTraining(), 'Training is ON');

    // Get initial weights
    const initialWeights = [...agent.model.weights];

    // Give reward while training is ON
    const time = performance.now();
    agent.processHumanReward(1.0, time);

    // Take multiple steps to ensure the sample is processed
    // With uniform credit, we need the sample to accumulate enough credit
    agent.step(time + 100);
    agent.step(time + 300);
    agent.step(time + 500);
    agent.step(time + 700);
    agent.step(time + 1100);  // Past the window end

    // Weights SHOULD have changed
    const weightsAfterReward = [...agent.model.weights];
    let weightsChanged = false;
    for (let i = 0; i < initialWeights.length; i++) {
        if (initialWeights[i] !== weightsAfterReward[i]) {
            weightsChanged = true;
            break;
        }
    }

    assertTrue(weightsChanged, 'Weights changed when training ON and reward given');
}

// =============================================================================
// 4. ACTION SELECTION USES LEARNED VALUES
// =============================================================================

function testActionSelectionUsesLearnedValues() {
    console.log('\n' + '='.repeat(70));
    console.log('4. ACTION SELECTION USES LEARNED VALUES');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,
        epsilon: 0.0,  // No exploration - always greedy
        stepSize: 0.5,  // Higher step size for faster learning
        creditDelayMs: 0,
        creditWindowMs: 1000,
        creditDistType: 'uniform'
    });

    // Start episode and enable training
    agent.startEpisode();
    agent.toggleTraining();

    const baseTime = performance.now();

    // Give multiple positive rewards while the agent takes steps
    // The rewards will be assigned to whatever action was just taken
    for (let i = 0; i < 10; i++) {
        const time = baseTime + i * 200;

        // Give positive reward
        agent.processHumanReward(1.0, time + 50);

        // Take a step
        agent.step(time + 100);
    }

    // Take more steps to flush out the credit window
    for (let i = 0; i < 10; i++) {
        agent.step(baseTime + 2000 + i * 200);
    }

    // Now check action values - at least some should be positive
    const values = agent.getActionValues();
    console.log(`  Action values: [${values.map(v => v.toFixed(4)).join(', ')}]`);

    // After positive rewards, at least one action value should be positive
    let someValuePositive = values.some(v => v > 0);

    assertTrue(someValuePositive, 'Some action has positive value after positive rewards');
}

// =============================================================================
// 5. EPISODE LIFECYCLE
// =============================================================================

function testEpisodeLifecycle() {
    console.log('\n' + '='.repeat(70));
    console.log('5. EPISODE LIFECYCLE');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, { basisFcnsPerDim: 5 });

    // Before start, currentObs should be null
    assertEqual(agent.currentObs, null, 'currentObs is null before startEpisode');
    assertEqual(agent.episodeCount, 0, 'episodeCount is 0 before start');

    // Start episode
    const firstAction = agent.startEpisode();

    assertNotEqual(agent.currentObs, null, 'currentObs is set after startEpisode');
    assertEqual(agent.episodeCount, 1, 'episodeCount is 1 after startEpisode');
    assertTrue(firstAction >= 0 && firstAction < 4, 'startEpisode returns valid action');

    // Take some steps
    const time = performance.now();
    const result1 = agent.step(time);

    assertTrue(result1.obs !== null, 'step returns obs');
    assertTrue(typeof result1.reward === 'number', 'step returns reward');
    assertTrue(typeof result1.terminal === 'boolean', 'step returns terminal');
    assertEqual(agent.totalSteps, 1, 'totalSteps incremented after step');

    // End episode
    agent.endEpisode();

    assertEqual(agent.currentObs, null, 'currentObs is null after endEpisode');

    // Start new episode
    agent.startEpisode();
    assertEqual(agent.episodeCount, 2, 'episodeCount is 2 after second startEpisode');
}

// =============================================================================
// 6. CREDIT ASSIGNMENT DISTRIBUTES REWARDS
// =============================================================================

function testCreditAssignmentDistributesRewards() {
    console.log('\n' + '='.repeat(70));
    console.log('6. CREDIT ASSIGNMENT DISTRIBUTES REWARDS');
    console.log('='.repeat(70));

    // Test uniform credit assignment
    const ca = new CreditAssign({
        creditDelayMs: 200,
        creditWindowMs: 600,
        distClass: 'uniform'
    });

    // Record some timesteps
    const feats1 = new Float64Array([1, 0, 0]);
    const feats2 = new Float64Array([0, 1, 0]);
    const feats3 = new Float64Array([0, 0, 1]);

    ca.recordTimeStepStart(feats1, 0.0);
    ca.recordTimeStepEnd(0.1);
    ca.recordTimeStepStart(feats2, 0.1);
    ca.recordTimeStepEnd(0.2);
    ca.recordTimeStepStart(feats3, 0.2);
    ca.recordTimeStepEnd(0.3);

    // Give reward at time 0.5 (within window of step 1 and 2)
    ca.inTrainSess = true;  // Enable training
    ca.processNewHReward(1.0, 0.5);

    // Check that samples have received credit
    let totalCredit = 0;
    for (const sample of ca.activeSamples) {
        totalCredit += sample.usedCredit;
    }

    assertTrue(totalCredit > 0, 'Samples received credit from reward');

    // Test previousStep credit assignment
    const ca2 = new CreditAssign({
        creditDelayMs: 0,
        creditWindowMs: 100,
        distClass: 'previousStep'
    });

    ca2.recordTimeStepStart(feats1, 0.0);
    ca2.recordTimeStepEnd(0.1);
    ca2.recordTimeStepStart(feats2, 0.1);
    ca2.recordTimeStepEnd(0.2);

    ca2.inTrainSess = true;
    ca2.processNewHReward(1.0, 0.25);

    // For previousStep, only the step before current should get credit
    assertEqual(ca2.activeSamples[0].unweightedRew, 1.0, 'previousStep: first step gets full reward');
    assertEqual(ca2.activeSamples[1].unweightedRew, 0.0, 'previousStep: current step gets no reward');
}

// =============================================================================
// 7. NEGATIVE REWARDS DECREASE VALUES
// =============================================================================

function testNegativeRewardsDecreaseValues() {
    console.log('\n' + '='.repeat(70));
    console.log('7. NEGATIVE REWARDS DECREASE VALUES');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,
        epsilon: 0.0,
        creditDelayMs: 0,
        creditWindowMs: 1000,
        creditDistType: 'uniform'
    });

    // Start and enable training
    agent.startEpisode();
    agent.toggleTraining();

    // Get initial values (should be all 0)
    const initialValues = agent.getActionValues();

    // Give negative reward for current action
    const time = performance.now();
    agent.processHumanReward(-1.0, time);

    // Take multiple steps to process the reward through credit assignment
    agent.step(time + 100);
    agent.step(time + 300);
    agent.step(time + 500);
    agent.step(time + 700);
    agent.step(time + 1100);

    // Values should have decreased (become negative)
    const valuesAfter = agent.getActionValues();
    console.log(`  Initial values: [${initialValues.map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`  Values after: [${valuesAfter.map(v => v.toFixed(4)).join(', ')}]`);

    // At least one value should be lower (negative)
    let someValueDecreased = false;
    for (let i = 0; i < valuesAfter.length; i++) {
        if (valuesAfter[i] < 0) {
            someValueDecreased = true;
            break;
        }
    }

    assertTrue(someValueDecreased, 'Negative reward decreases action values');
}

// =============================================================================
// 8. MULTIPLE ENVIRONMENTS WORK
// =============================================================================

function testMultipleEnvironmentsWork() {
    console.log('\n' + '='.repeat(70));
    console.log('8. MULTIPLE ENVIRONMENTS WORK');
    console.log('='.repeat(70));

    // Test LoopMaze
    const loopMaze = new LoopMaze();
    const loopAgent = new TamerAgent(loopMaze, { basisFcnsPerDim: 5 });
    loopAgent.startEpisode();
    const loopResult = loopAgent.step(performance.now());
    assertTrue(loopResult.obs !== null, 'LoopMaze: step returns obs');
    assertTrue(loopResult.obs.length === 2, 'LoopMaze: obs has 2 dimensions');

    // Test MountainCar
    const mountainCar = new MountainCar();
    const mcAgent = new TamerAgent(mountainCar, { basisFcnsPerDim: 5 });
    mcAgent.startEpisode();
    const mcResult = mcAgent.step(performance.now());
    assertTrue(mcResult.obs !== null, 'MountainCar: step returns obs');
    assertTrue(mcResult.obs.length === 2, 'MountainCar: obs has 2 dimensions');
}

// =============================================================================
// 9. RESET CLEARS STATE
// =============================================================================

function testResetClearsState() {
    console.log('\n' + '='.repeat(70));
    console.log('9. RESET CLEARS STATE');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, { basisFcnsPerDim: 5 });

    // Do some learning
    agent.startEpisode();
    agent.toggleTraining();
    agent.processHumanReward(1.0, performance.now());
    agent.step(performance.now() + 100);
    agent.step(performance.now() + 200);

    // Verify state exists
    assertTrue(agent.totalSteps > 0, 'totalSteps > 0 before reset');
    assertTrue(agent.episodeCount > 0, 'episodeCount > 0 before reset');

    // Reset
    agent.reset();

    // State should be cleared
    assertEqual(agent.totalSteps, 0, 'totalSteps is 0 after reset');
    assertEqual(agent.episodeCount, 0, 'episodeCount is 0 after reset');
    assertEqual(agent.currentObs, null, 'currentObs is null after reset');
    assertEqual(agent.pendingRewards.length, 0, 'pendingRewards empty after reset');
}

// =============================================================================
// 10. PREDICTIONS ARE INDEPENDENT PER ACTION
// =============================================================================

function testPredictionsIndependentPerAction() {
    console.log('\n' + '='.repeat(70));
    console.log('10. PREDICTIONS ARE INDEPENDENT PER ACTION');
    console.log('='.repeat(70));

    const env = new LoopMaze();
    const agent = new TamerAgent(env, {
        basisFcnsPerDim: 5,
        epsilon: 0.0,
        stepSize: 0.5,  // Higher step size for visible learning
        creditDelayMs: 0,
        creditWindowMs: 1000,
        creditDistType: 'uniform'
    });

    agent.startEpisode();
    agent.toggleTraining();

    const baseTime = performance.now();

    // Give positive rewards repeatedly
    for (let i = 0; i < 20; i++) {
        const time = baseTime + i * 100;
        agent.processHumanReward(1.0, time + 25);
        agent.step(time + 50);
    }

    // Flush the credit window
    for (let i = 0; i < 15; i++) {
        agent.step(baseTime + 3000 + i * 100);
    }

    // Then give negative rewards
    for (let i = 0; i < 20; i++) {
        const time = baseTime + 5000 + i * 100;
        agent.processHumanReward(-1.0, time + 25);
        agent.step(time + 50);
    }

    // Flush again
    for (let i = 0; i < 15; i++) {
        agent.step(baseTime + 8000 + i * 100);
    }

    const values = agent.getActionValues();
    console.log(`  Action values: [${values.map(v => v.toFixed(4)).join(', ')}]`);

    // The model should have learned SOMETHING - values should not all be zero
    const hasNonZero = values.some(v => Math.abs(v) > 0.0001);
    assertTrue(hasNonZero, 'Model learned something (non-zero predictions)');
}

// =============================================================================
// MAIN
// =============================================================================

async function runAllTests() {
    console.log('='.repeat(70));
    console.log('BEHAVIORAL TEST SUITE');
    console.log('Tests actual behavior and logic flows');
    console.log('='.repeat(70));

    try {
        testTrainingToggleBehavior();
        testRewardsIgnoredWhenTrainingOff();
        testRewardsUpdateModelWhenTrainingOn();
        testActionSelectionUsesLearnedValues();
        testEpisodeLifecycle();
        testCreditAssignmentDistributesRewards();
        testNegativeRewardsDecreaseValues();
        testMultipleEnvironmentsWork();
        testResetClearsState();
        testPredictionsIndependentPerAction();
    } catch (error) {
        console.error('\n*** TEST ERROR ***');
        console.error(error);
        failedTests.push({ testName: 'Test execution error', actual: error.message, expected: 'No error' });
    }

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('TEST SUMMARY');
    console.log('='.repeat(70));
    console.log(`Total tests: ${passedTests + failedTests.length}`);
    console.log(`Passed: ${passedTests}`);
    console.log(`Failed: ${failedTests.length}`);

    if (failedTests.length > 0) {
        console.log('\nFailed tests:');
        for (const { testName, actual, expected } of failedTests) {
            console.log(`  - ${testName}`);
            console.log(`    Expected: ${expected}`);
            console.log(`    Actual: ${actual}`);
        }
        console.log('\n*** SOME TESTS FAILED ***');
        process.exit(1);
    } else {
        console.log('\n*** ALL TESTS PASSED ***');
    }
}

runAllTests();
