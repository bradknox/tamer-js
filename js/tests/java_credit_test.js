/**
 * Exact replication of Java CreditAssign.main() test case
 *
 * This runs the EXACT test from CreditAssign.java lines 606-663
 * to verify the JS port produces identical results.
 */

import { CreditAssign } from '../core/CreditAssign.js';

console.log('======================================================================');
console.log('JAVA CREDIT ASSIGN TEST CASE REPLICATION');
console.log('Exact copy of CreditAssign.java main() test');
console.log('======================================================================\n');

// CREATE CreditAssignParamVec - EXACT from Java lines 607-615
const credA = new CreditAssign({
    distClass: 'uniform',
    creditDelayMs: 200,      // creditDelay = 0.2
    creditWindowMs: 600,     // windowSize = 0.6
    extrapolateFutureRew: true,
    delayWtedIndivRew: false,
    noUpdateWhenNoRew: true
});

// credA.setInTrainSess(0.0, true) - EXACT from Java line 616
credA.setInTrainSess(0.0, true);

console.log('Test 1: 10 steps with reward at step 9 (t=1.8s)');
console.log('----------------------------------------------------------------------');

// EXACT loop from Java lines 618-627
for (let i = 0; i < 10; i++) {
    const feats = new Float64Array([i, i + 10]);
    const time = 0.2 * i;

    credA.recordTimeStepEnd(time);
    credA.recordTimeStepStart(feats, time, null, null);

    // Reward of 1.0 at t=1.8 (i=9), 0 otherwise - EXACT from Java line 623
    const reward = (Math.abs(time - 1.8) < 0.001) ? 1 : 0;
    credA.processNewHReward(reward, time);

    const samplesForUpdate = credA.processSamplesAndRemoveFinished(time, true);

    console.log(`\nStep ${i} (t=${time.toFixed(1)}s):`);
    console.log(`  Reward given: ${reward}`);
    console.log(`  Samples for update: ${samplesForUpdate.length}`);
    for (const s of samplesForUpdate) {
        console.log(`    label=${s.label.toFixed(4)}, usedCredit=${s.usedCredit.toFixed(4)}, weight=${s.weight}`);
    }
    console.log(`  Active samples: ${credA.activeSamples.length}`);
    for (let j = 0; j < credA.activeSamples.length; j++) {
        const s = credA.activeSamples[j];
        console.log(`    [${j}] unweightedRew=${s.unweightedRew.toFixed(4)}, label=${s.label.toFixed(4)}, usedCredit=${s.usedCredit.toFixed(4)}`);
    }
}

console.log('\n======================================================================');
console.log('Test 2: Journal paper example (steps at 3.25s and 3.45s)');
console.log('----------------------------------------------------------------------');

// Clear and set up for journal paper example - EXACT from Java lines 643-653
credA.clearHistory();

const feats2 = new Float64Array([10, 20]);

// Step starting at 3.25s
credA.recordTimeStepStart(feats2, 3.25, null, null);
console.log('Step A started at t=3.25s');

// Reward at 3.4s
credA.processNewHReward(2.0, 3.4);
console.log('Reward +2.0 given at t=3.4s');

// Step ends at 3.45s, new step starts
credA.recordTimeStepEnd(3.45);
credA.recordTimeStepStart(feats2, 3.45, null, null);
console.log('Step A ended at t=3.45s, Step B started');

// Another reward at 3.85s
credA.processNewHReward(2.0, 3.85);
console.log('Reward +2.0 given at t=3.85s');

// Process at 3.9s
const creditedSamples2 = credA.processSamplesAndRemoveFinished(3.9, true);

console.log('\nAt t=3.9s:');
console.log('Active samples:');
for (let i = 0; i < credA.activeSamples.length; i++) {
    const s = credA.activeSamples[i];
    const step = credA.timeStepsInWindow[i];
    console.log(`  [${i}] start=${step.startTime.toFixed(2)}s, end=${step.endTime === Infinity ? 'ongoing' : step.endTime.toFixed(2) + 's'}`);
    console.log(`       unweightedRew=${s.unweightedRew.toFixed(4)}, label=${s.label.toFixed(4)}, usedCredit=${s.usedCredit.toFixed(4)}`);
}

console.log('\nCredited samples for update:');
for (const s of creditedSamples2) {
    console.log(`  label=${s.label.toFixed(4)}, usedCredit=${s.usedCredit.toFixed(4)}`);
}

// Final reward at 4.1s
credA.processNewHReward(2.0, 4.1);
console.log('\nReward +2.0 given at t=4.1s');

// Process at 4.3s
const finalSamples = credA.processSamplesAndRemoveFinished(4.3, true);
console.log('\nAt t=4.3s:');
console.log('Final credited samples:');
for (const s of finalSamples) {
    console.log(`  label=${s.label.toFixed(4)}, usedCredit=${s.usedCredit.toFixed(4)}`);
}

console.log('\n======================================================================');
console.log('EXPECTED BEHAVIOR (from Java):');
console.log('----------------------------------------------------------------------');
console.log('With uniform credit, creditDelay=0.2, windowSize=0.6:');
console.log('- Window covers 0.2-0.8 seconds before reward time');
console.log('- Steps fully in window get proportional credit');
console.log('- Extrapolation divides by usedCredit when < 1.0');
console.log('');
console.log('For journal example with 3 rewards of +2.0 each = +6.0 total:');
console.log('- Step A (3.25-3.45s) should eventually get its fair share');
console.log('- Step B (3.45s-ongoing) should get its fair share');
