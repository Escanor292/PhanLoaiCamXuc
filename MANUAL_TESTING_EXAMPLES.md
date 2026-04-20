# Manual Testing Examples - Expected Outputs

This document shows example outputs for manual testing to help verify the system is working correctly.

---

## Example 1: Strong Positive Emotions

**Input:**
```
I absolutely love this product! It exceeded all my expectations and made me so happy!
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "I absolutely love this product! It exceeded all my expectations and made me so happy!"

----------------------------------------------------------------------

Predicted Emotions (4):
----------------------------------------------------------------------
  joy             [████████████████████████████████████░░░░] 0.920
  love            [██████████████████████████████░░░░░░░░░░] 0.850
  excited         [████████████████████████░░░░░░░░░░░░░░░] 0.720
  trust           [████████████████████░░░░░░░░░░░░░░░░░░░] 0.650

======================================================================
```

**Validation:**
- ✓ Should detect "joy" (explicit: "happy")
- ✓ Should detect "love" (explicit: "love")
- ✓ Should detect "excited" (implicit: "exceeded expectations")
- ✓ Should detect "trust" (implicit: positive experience)

---

## Example 2: Strong Negative Emotions

**Input:**
```
This is terrible. I'm very disappointed and angry with the service.
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "This is terrible. I'm very disappointed and angry with the service."

----------------------------------------------------------------------

Predicted Emotions (3):
----------------------------------------------------------------------
  anger           [████████████████████████████████░░░░░░░░] 0.880
  disappointed    [██████████████████████████████████░░░░░░] 0.910
  disgust         [██████████████████████░░░░░░░░░░░░░░░░░] 0.720

======================================================================
```

**Validation:**
- ✓ Should detect "anger" (explicit: "angry")
- ✓ Should detect "disappointed" (explicit: "disappointed")
- ✓ Should detect "disgust" (implicit: "terrible")

---

## Example 3: Mixed Emotions

**Input:**
```
I'm excited but also worried about the new project. Hope it goes well!
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "I'm excited but also worried about the new project. Hope it goes well!"

----------------------------------------------------------------------

Predicted Emotions (3):
----------------------------------------------------------------------
  excited         [████████████████████████████░░░░░░░░░░░░] 0.820
  worried         [██████████████████████████░░░░░░░░░░░░░░] 0.750
  anticipation    [████████████████████░░░░░░░░░░░░░░░░░░░] 0.680

======================================================================
```

**Validation:**
- ✓ Should detect "excited" (explicit: "excited")
- ✓ Should detect "worried" (explicit: "worried")
- ✓ Should detect "anticipation" (explicit: "hope", future-oriented)
- ✓ Demonstrates multi-label capability (positive + negative simultaneously)

---

## Example 4: Neutral/Calm

**Input:**
```
The product works as described. No issues so far.
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "The product works as described. No issues so far."

----------------------------------------------------------------------

Predicted Emotions (2):
----------------------------------------------------------------------
  calm            [██████████████████████░░░░░░░░░░░░░░░░░] 0.680
  trust           [████████████████████░░░░░░░░░░░░░░░░░░░] 0.620

======================================================================
```

**Validation:**
- ✓ Should detect "calm" (implicit: neutral, satisfied tone)
- ✓ Should detect "trust" (implicit: meets expectations)
- ✓ Lower confidence scores are normal for neutral text

---

## Example 5: Vietnamese Text

**Input:**
```
Tôi rất vui và hạnh phúc với sản phẩm này!
```

**Translation:** "I'm very happy and pleased with this product!"

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "Tôi rất vui và hạnh phúc với sản phẩm này!"

----------------------------------------------------------------------

Predicted Emotions (2-3):
----------------------------------------------------------------------
  joy             [████████████████████░░░░░░░░░░░░░░░░░░░] 0.650
  love            [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 0.550

======================================================================
```

**Validation:**
- ✓ Should attempt to detect positive emotions
- ⚠ Confidence scores may be lower than English text
- ⚠ bert-base-uncased is English-focused (expected limitation)

---

## Example 6: Edge Case - Very Short Text

**Input:**
```
ok
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "ok"

----------------------------------------------------------------------

Predicted Emotions (1):
----------------------------------------------------------------------
  calm            [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.520

======================================================================
```

**Validation:**
- ✓ Should handle gracefully without errors
- ✓ May detect "calm" with low-to-moderate confidence
- ⚠ Low confidence is expected due to minimal context

---

## Example 7: Edge Case - Only Punctuation

**Input:**
```
!!!
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "!!!"

----------------------------------------------------------------------

Predicted Emotions: None
(No emotions exceeded the confidence threshold)

======================================================================
```

**Validation:**
- ✓ Should handle gracefully without errors
- ✓ Should return empty emotions or very low confidence scores
- ✓ Demonstrates proper handling of edge cases

---

## Example 8: Edge Case - Emojis

**Input:**
```
😊😍🎉
```

**Expected Output:**
```
======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "😊😍🎉"

----------------------------------------------------------------------

Predicted Emotions (2-3):
----------------------------------------------------------------------
  joy             [████████████████░░░░░░░░░░░░░░░░░░░░░░░] 0.580
  love            [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 0.540
  excited         [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.510

======================================================================
```

**Validation:**
- ✓ Should attempt to interpret emojis
- ⚠ BERT tokenizer may not preserve emoji meaning perfectly
- ⚠ Confidence scores may vary depending on tokenization

---

## Example 9: Batch Prediction

**Input:**
```python
texts = [
    "I love this!",
    "This is terrible.",
    "Feeling calm and peaceful."
]
```

**Expected Output:**
```
Batch prediction successful!
Processed 3 comments in batch mode

1. "I love this!"
   Emotions: joy, love, excited

2. "This is terrible."
   Emotions: anger, disappointed, disgust

3. "Feeling calm and peaceful."
   Emotions: calm

✓ Batch prediction demonstrates efficient multi-comment processing
```

**Validation:**
- ✓ Should process all comments without errors
- ✓ Should return results in same order as input
- ✓ Each result should have appropriate emotions

---

## Interpreting Results

### Confidence Scores

- **0.9 - 1.0:** Very high confidence (strong emotion clearly expressed)
- **0.7 - 0.9:** High confidence (emotion clearly present)
- **0.5 - 0.7:** Moderate confidence (emotion likely present)
- **0.3 - 0.5:** Low confidence (emotion possibly present, below threshold)
- **0.0 - 0.3:** Very low confidence (emotion unlikely)

### Visual Bars

```
[████████████████████████████████████████] 1.000  (Full bar = 100%)
[████████████████████████████████░░░░░░░░] 0.800  (80% filled)
[████████████████████░░░░░░░░░░░░░░░░░░░] 0.500  (50% filled - threshold)
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.250  (25% filled - below threshold)
```

### Success Indicators

✓ **Test Passed:** At least one expected emotion detected  
⚠ **Acceptable:** Reasonable prediction even if not exact match  
✗ **Test Failed:** System error or completely incorrect prediction

---

## Common Observations

### 1. Multiple Emotions
The system can detect multiple emotions simultaneously:
```
Input: "I'm happy but also worried"
Output: joy (0.75), worried (0.68)
```

### 2. Confidence Variation
Confidence scores vary based on:
- Explicitness of emotion words
- Amount of context provided
- Clarity of emotional expression

### 3. Threshold Effect
Default threshold is 0.5:
- Emotions ≥ 0.5 are included in predictions
- Emotions < 0.5 are excluded (but visible in "all scores" view)

### 4. Language Limitation
Vietnamese text may have:
- Lower confidence scores
- Less accurate predictions
- This is expected with bert-base-uncased

---

## Testing Checklist

When performing manual testing, verify:

- [ ] System loads model without errors
- [ ] Positive emotions are detected correctly
- [ ] Negative emotions are detected correctly
- [ ] Mixed emotions can be detected simultaneously
- [ ] Neutral/calm states are identified
- [ ] Vietnamese text is processed (even if accuracy is limited)
- [ ] Edge cases are handled gracefully
- [ ] Batch prediction works correctly
- [ ] Confidence scores are in range [0, 1]
- [ ] No system crashes or errors

---

## Troubleshooting

### Issue: Model not found
```
✗ Error: Model checkpoint not found
```
**Solution:** Train the model first: `python train.py`

### Issue: Low confidence scores
```
All emotions below threshold
```
**Possible causes:**
- Very short or ambiguous text
- Neutral emotional content
- Text in unsupported language

**Solution:** Try more explicit emotional language

### Issue: Unexpected emotions detected
```
Predicted emotions don't match expected
```
**Possible causes:**
- Subjective interpretation differences
- Model trained on different data distribution
- Ambiguous or sarcastic text

**Solution:** Review confidence scores; consider if prediction is reasonable

---

## Next Steps

After manual testing:

1. ✓ Review MANUAL_TESTING_REPORT.md for comprehensive analysis
2. ✓ Check TASK_11.2_SUMMARY.md for task completion details
3. ✓ Run automated tests: `python manual_testing.py`
4. ✓ Run quick tests: `python run_manual_tests.py`
5. → Consider recommendations for production deployment
6. → Fine-tune model if needed for specific use cases

---

**Document Purpose:** Provide example outputs for manual testing validation  
**Last Updated:** 2026-04-20  
**Related Files:** MANUAL_TESTING_REPORT.md, manual_testing.py, run_manual_tests.py
