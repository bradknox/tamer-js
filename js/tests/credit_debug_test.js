/**
 * Credit Assignment Diagnostic Test
 *
 * Simulates realistic TAMER training with delayed feedback
 * to verify credit is going to the correct timesteps.
 */

import { CreditAssign } from '../core/CreditAssign.js';

console.log('======================================================================');
console.log('CREDIT ASSIGNMENT DIAGNOSTIC TEST');
console.log('======================================================================\n');

/**
 * Test 1: Verify credit goes to correct timesteps with uniform distribution
 *
 * Setup:
 * - creditDelay = 200ms (0.2s)
 * - creditWindow = 600ms (0.6s)
 * - So credit window is [T - 0.8s, T - 0.2s] for reward at time T
 *
 * Scenario:
 * - Steps occur every 200ms (0.2s)
 * - Step 0: t=0.0 to t=0.2
 * - Step 1: t=0.2 to t=0.4
 * - Step 2: t=0.4 to t=0.6
 * - Step 3: t=0.6 to t=0.8
 * - Step 4: t=0.8 to t=1.0
 *
 * Reward given at t=1.0:
 * - Credit window: [1.0 - 0.8, 1.0 - 0.2] = [0.2, 0.8]
 * - Step 0 (0.0-0.2): entirely before window, credit = 0
 * - Step 1 (0.2-0.4): entirely in window, credit > 0
 * - Step 2 (0.4-0.6): entirely in window, credit > 0
 * - Step 3 (0.6-0.8): entirely in window, credit > 0
 * - Step 4 (0.8-1.0): entirely after window start, credit = 0
 */

console.log('TEST 1: Uniform Credit Assignment with 200ms Steps');
console.log('----------------------------------------------------------------------');

const credAssign = new CreditAssign({
    distClass: 'uniform',
    creditDelayMs: 200,
    creditWindowMs: 600,
    extrapolateFutureRew: true
});

// Enable training
credAssign.setInTrainSess(0, true);

// Simulate 5 steps, each 0.2 seconds apart
const stepDuration = 0.2;  // seconds
const steps = [];

for (let i = 0; i < 5; i++) {
    const startTime = i * stepDuration;
    const endTime = (i + 1) * stepDuration;

    // Record step end (except for first step)
    if (i > 0) {
        credAssign.recordTimeStepEnd(startTime);
    }

    // Record step start
    const feats = new Float64Array([i, i + 1]);  // dummy features
    credAssign.recordTimeStepStart(feats, startTime, [i], i);

    steps.push({ startTime, endTime, feats });
    console.log(`  Step ${i}: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`);
}

// End the last step
credAssign.recordTimeStepEnd(1.0);
console.log('');

// Give reward at t=1.0
const rewardTime = 1.0;
console.log(`Reward given at t=${rewardTime}s`);
console.log(`Credit window: [${(rewardTime - 0.8).toFixed(2)}s, ${(rewardTime - 0.2).toFixed(2)}s]`);
console.log('');

// Before processing reward, check the active samples
console.log('Active samples before reward:');
for (let i = 0; i < credAssign.activeSamples.length; i++) {
    const sample = credAssign.activeSamples[i];
    const step = credAssign.timeStepsInWindow[i];
    console.log(`  Sample ${i}: start=${step.startTime.toFixed(2)}s, end=${step.endTime.toFixed(2)}s, unweightedRew=${sample.unweightedRew}`);
}
console.log('');

// Process the reward
credAssign.processNewHReward(1.0, rewardTime);

// Check credit distribution
console.log('Credit distribution after reward:');
let totalCredit = 0;
for (let i = 0; i < credAssign.activeSamples.length; i++) {
    const sample = credAssign.activeSamples[i];
    const step = credAssign.timeStepsInWindow[i];
    const credit = sample.unweightedRew;  // For a reward of 1.0, unweightedRew equals credit
    totalCredit += credit;
    console.log(`  Step ${i} (${step.startTime.toFixed(2)}s-${step.endTime.toFixed(2)}s): credit=${credit.toFixed(4)}`);
}
console.log(`  Total credit: ${totalCredit.toFixed(4)} (should be ~1.0 for contiguous steps)`);
console.log('');

// Verify expected behavior
const step0Credit = credAssign.activeSamples[0].unweightedRew;
const step1Credit = credAssign.activeSamples[1].unweightedRew;
const step2Credit = credAssign.activeSamples[2].unweightedRew;
const step3Credit = credAssign.activeSamples[3].unweightedRew;
const step4Credit = credAssign.activeSamples[4].unweightedRew;

console.log('Verification:');
console.log(`  Step 0 (before window): ${step0Credit === 0 ? '✓ PASS' : '✗ FAIL'} credit=${step0Credit.toFixed(4)} (expected 0)`);
console.log(`  Step 1 (in window): ${step1Credit > 0 ? '✓ PASS' : '✗ FAIL'} credit=${step1Credit.toFixed(4)} (expected > 0)`);
console.log(`  Step 2 (in window): ${step2Credit > 0 ? '✓ PASS' : '✗ FAIL'} credit=${step2Credit.toFixed(4)} (expected > 0)`);
console.log(`  Step 3 (in window): ${step3Credit > 0 ? '✓ PASS' : '✗ FAIL'} credit=${step3Credit.toFixed(4)} (expected > 0)`);
console.log(`  Step 4 (after window): ${step4Credit === 0 ? '✓ PASS' : '✗ FAIL'} credit=${step4Credit.toFixed(4)} (expected 0)`);

console.log('\n======================================================================');
console.log('TEST 2: Reward Given During a Step (More Realistic)');
console.log('======================================================================\n');

/**
 * More realistic: reward comes in the middle of step execution
 *
 * - Steps occur every 800ms (default for LoopMaze)
 * - Reward given 500ms after an action (in the middle of the credit window)
 */

const credAssign2 = new CreditAssign({
    distClass: 'uniform',
    creditDelayMs: 200,
    creditWindowMs: 600,
    extrapolateFutureRew: true
});

credAssign2.setInTrainSess(0, true);

// Step 0: 0.0 - 0.8
credAssign2.recordTimeStepStart(new Float64Array([0]), 0.0, [0], 0);
credAssign2.recordTimeStepEnd(0.8);

// Step 1: 0.8 - 1.6
credAssign2.recordTimeStepStart(new Float64Array([1]), 0.8, [1], 1);
credAssign2.recordTimeStepEnd(1.6);

// Step 2: 1.6 - 2.4
credAssign2.recordTimeStepStart(new Float64Array([2]), 1.6, [2], 2);
// End not recorded yet (current step)

console.log('Steps (800ms each):');
console.log('  Step 0: 0.0s - 0.8s');
console.log('  Step 1: 0.8s - 1.6s');
console.log('  Step 2: 1.6s - ongoing');
console.log('');

// Reward at t=2.1 (500ms after step 2 started)
const rewardTime2 = 2.1;
console.log(`Reward given at t=${rewardTime2}s`);
console.log(`Credit window: [${(rewardTime2 - 0.8).toFixed(2)}s, ${(rewardTime2 - 0.2).toFixed(2)}s] = [1.3s, 1.9s]`);
console.log('');

// Process reward
credAssign2.processNewHReward(1.0, rewardTime2);

console.log('Credit distribution:');
for (let i = 0; i < credAssign2.activeSamples.length; i++) {
    const sample = credAssign2.activeSamples[i];
    const step = credAssign2.timeStepsInWindow[i];
    const credit = sample.unweightedRew;
    console.log(`  Step ${i} (${step.startTime.toFixed(2)}s-${step.endTime === Infinity ? 'ongoing' : step.endTime.toFixed(2) + 's'}): credit=${credit.toFixed(4)}`);
}
console.log('');

// For window [1.3, 1.9]:
// Step 0 (0.0-0.8): entirely before window, credit = 0
// Step 1 (0.8-1.6): overlaps [1.3, 1.6], partial credit
// Step 2 (1.6-ongoing): overlaps [1.6, 1.9], partial credit
console.log('Expected:');
console.log('  Step 0 (0.0-0.8): Should be 0 (entirely before window)');
console.log('  Step 1 (0.8-1.6): Should be ~0.5 (partial overlap with window)');
console.log('  Step 2 (1.6-ongoing): Should be ~0.5 (partial overlap with window)');

console.log('\n======================================================================');
console.log('TEST 3: What Happens With "Immediate" Feedback');
console.log('======================================================================\n');

/**
 * User gives feedback immediately after an action (within 200ms delay)
 * With 200ms delay, feedback given < 200ms after action gets NO credit!
 */

const credAssign3 = new CreditAssign({
    distClass: 'uniform',
    creditDelayMs: 200,
    creditWindowMs: 600,
    extrapolateFutureRew: true
});

credAssign3.setInTrainSess(0, true);

// Step 0: 0.0 - 0.8
credAssign3.recordTimeStepStart(new Float64Array([0]), 0.0, [0], 0);
// Step not ended yet

// Reward given immediately at t=0.1 (100ms after action)
const rewardTime3 = 0.1;
console.log(`Step 0 started at t=0.0s`);
console.log(`Reward given at t=${rewardTime3}s (100ms after action)`);
console.log(`Credit window: [${(rewardTime3 - 0.8).toFixed(2)}s, ${(rewardTime3 - 0.2).toFixed(2)}s]`);
console.log('');

credAssign3.processNewHReward(1.0, rewardTime3);

const immediateCredit = credAssign3.activeSamples[0].unweightedRew;
console.log(`Credit to Step 0: ${immediateCredit.toFixed(4)}`);
console.log(`This is ${immediateCredit === 0 ? 'ZERO - reward too fast!' : 'non-zero'}`);
console.log('');
console.log('NOTE: With 200ms delay, feedback given within 200ms of an action');
console.log('      will NOT be credited to that action! This is by design - it');
console.log('      assumes human reaction time is at least 200ms.');

console.log('\n======================================================================');
console.log('SUMMARY');
console.log('======================================================================\n');

console.log('The credit assignment appears to be working correctly if:');
console.log('1. Steps entirely within the window get credit');
console.log('2. Steps entirely outside the window get 0 credit');
console.log('3. Steps partially in the window get partial credit');
console.log('');
console.log('Common issues that could cause "wrong timestep" credit:');
console.log('1. Feedback given too quickly (< 200ms) gets no credit to current step');
console.log('2. Feedback given too slowly (> 800ms) misses the intended step entirely');
console.log('3. Step duration affects which steps fall in the window');
