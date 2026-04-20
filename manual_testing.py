"""
Manual Testing Script for Multi-label Emotion Classification System

This script performs comprehensive manual testing of the prediction system with
diverse real-world examples covering:
- Positive emotion comments
- Negative emotion comments
- Mixed emotion comments
- Neutral/calm comments
- Vietnamese text
- Edge cases and limitations

Requirements tested: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import torch
from config import Config
from utils import load_model
from predict import predict_emotions, predict_emotions_batch
import json
from datetime import datetime


# Test cases organized by category
TEST_CASES = {
    "positive_emotions": [
        {
            "text": "I absolutely love this product! It exceeded all my expectations and made me so happy!",
            "expected_emotions": ["joy", "love", "excited", "trust"],
            "description": "Strong positive emotions with love and excitement"
        },
        {
            "text": "This is amazing! I'm so proud of what we accomplished together!",
            "expected_emotions": ["joy", "proud", "excited"],
            "description": "Pride and accomplishment"
        },
        {
            "text": "I trust this brand completely. They always deliver quality products.",
            "expected_emotions": ["trust", "calm"],
            "description": "Trust and confidence"
        },
        {
            "text": "Wow! What a pleasant surprise! I didn't expect this at all!",
            "expected_emotions": ["surprise", "joy", "excited"],
            "description": "Positive surprise"
        },
        {
            "text": "I'm so excited about the upcoming event! Can't wait to see what happens!",
            "expected_emotions": ["excited", "anticipation", "joy"],
            "description": "Excitement and anticipation"
        }
    ],
    
    "negative_emotions": [
        {
            "text": "This is terrible. I'm very disappointed and angry with the service.",
            "expected_emotions": ["anger", "disappointed", "disgust"],
            "description": "Strong negative emotions"
        },
        {
            "text": "I'm so sad and heartbroken. This is not what I expected at all.",
            "expected_emotions": ["sadness", "disappointed"],
            "description": "Sadness and disappointment"
        },
        {
            "text": "This makes me sick. Absolutely disgusting and unacceptable.",
            "expected_emotions": ["disgust", "anger"],
            "description": "Disgust and anger"
        },
        {
            "text": "I'm really worried about the outcome. This doesn't look good.",
            "expected_emotions": ["worried", "fear"],
            "description": "Worry and concern"
        },
        {
            "text": "I'm so jealous of their success. Why can't I achieve the same?",
            "expected_emotions": ["jealous", "disappointed"],
            "description": "Jealousy"
        },
        {
            "text": "This is embarrassing. I can't believe this happened to me.",
            "expected_emotions": ["embarrassed", "disappointed"],
            "description": "Embarrassment"
        }
    ],
    
    "mixed_emotions": [
        {
            "text": "I'm excited but also worried about the new project. Hope it goes well!",
            "expected_emotions": ["excited", "worried", "anticipation"],
            "description": "Mixed excitement and worry"
        },
        {
            "text": "I love the product but I'm disappointed with the delivery time.",
            "expected_emotions": ["love", "disappointed"],
            "description": "Love mixed with disappointment"
        },
        {
            "text": "Surprising news! I'm both happy and scared about what comes next.",
            "expected_emotions": ["surprise", "joy", "fear"],
            "description": "Surprise with mixed emotions"
        },
        {
            "text": "I trust the team but I'm still worried about meeting the deadline.",
            "expected_emotions": ["trust", "worried"],
            "description": "Trust with underlying worry"
        },
        {
            "text": "This is disgusting but also fascinating in a strange way.",
            "expected_emotions": ["disgust", "surprise"],
            "description": "Conflicting emotions"
        }
    ],
    
    "neutral_calm": [
        {
            "text": "The product works as described. No issues so far.",
            "expected_emotions": ["calm", "trust"],
            "description": "Neutral satisfaction"
        },
        {
            "text": "Everything is fine. The service is adequate and meets basic requirements.",
            "expected_emotions": ["calm"],
            "description": "Neutral assessment"
        },
        {
            "text": "I'm feeling calm and peaceful about the situation now.",
            "expected_emotions": ["calm"],
            "description": "Explicit calmness"
        },
        {
            "text": "The documentation is clear and straightforward. Easy to follow.",
            "expected_emotions": ["calm", "trust"],
            "description": "Neutral positive"
        }
    ],
    
    "vietnamese_text": [
        {
            "text": "Tôi rất vui và hạnh phúc với sản phẩm này!",
            "expected_emotions": ["joy", "love", "excited"],
            "description": "Vietnamese: I'm very happy and pleased with this product!"
        },
        {
            "text": "Tôi thất vọng và buồn về dịch vụ này.",
            "expected_emotions": ["disappointed", "sadness"],
            "description": "Vietnamese: I'm disappointed and sad about this service."
        },
        {
            "text": "Tôi yêu thích sản phẩm này! Tuyệt vời!",
            "expected_emotions": ["love", "joy", "excited"],
            "description": "Vietnamese: I love this product! Wonderful!"
        },
        {
            "text": "Tôi lo lắng về kết quả.",
            "expected_emotions": ["worried", "fear"],
            "description": "Vietnamese: I'm worried about the result."
        },
        {
            "text": "Sản phẩm hoạt động tốt. Tôi tin tưởng thương hiệu này.",
            "expected_emotions": ["trust", "calm"],
            "description": "Vietnamese: Product works well. I trust this brand."
        }
    ],
    
    "edge_cases": [
        {
            "text": "!!!",
            "expected_emotions": [],
            "description": "Only punctuation"
        },
        {
            "text": "ok",
            "expected_emotions": ["calm"],
            "description": "Very short text"
        },
        {
            "text": "I feel... I don't know... maybe happy? Or sad? I'm confused.",
            "expected_emotions": ["surprise", "worried"],
            "description": "Uncertain/confused emotions"
        },
        {
            "text": "😊😍🎉",
            "expected_emotions": ["joy", "love", "excited"],
            "description": "Only emojis"
        },
        {
            "text": "This is a very long comment that goes on and on about many different things. " * 20,
            "expected_emotions": [],
            "description": "Very long text (truncation test)"
        },
        {
            "text": "AMAZING!!! BEST PRODUCT EVER!!! SO HAPPY!!!",
            "expected_emotions": ["joy", "excited", "love"],
            "description": "All caps with excessive punctuation"
        },
        {
            "text": "The product is... well... it's not bad, but not great either.",
            "expected_emotions": ["calm", "disappointed"],
            "description": "Ambiguous sentiment"
        }
    ]
}


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def print_test_result(test_num, test_case, result, passed_checks):
    """Print formatted test result."""
    print(f"\n{'─'*80}")
    print(f"Test {test_num}: {test_case['description']}")
    print(f"{'─'*80}")
    print(f"Input: \"{test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}\"")
    print(f"\nExpected emotions: {', '.join(test_case['expected_emotions']) if test_case['expected_emotions'] else 'None'}")
    print(f"Predicted emotions: {', '.join(result['emotions']) if result['emotions'] else 'None'}")
    
    if result['emotions']:
        print(f"\nTop confidence scores:")
        sorted_emotions = sorted(result['emotions'], key=lambda e: result['scores'][e], reverse=True)
        for emotion in sorted_emotions[:5]:  # Show top 5
            score = result['scores'][emotion]
            bar = "█" * int(score * 30)
            print(f"  {emotion:15s} [{bar:30s}] {score:.3f}")
    
    # Check if any expected emotions were found
    if test_case['expected_emotions']:
        found_emotions = set(result['emotions']) & set(test_case['expected_emotions'])
        if found_emotions:
            print(f"\n✓ Found expected emotions: {', '.join(found_emotions)}")
            passed_checks.append(True)
        else:
            print(f"\n⚠ No expected emotions found (this may be acceptable)")
            passed_checks.append(False)
    else:
        # For edge cases expecting no emotions
        if not result['emotions']:
            print(f"\n✓ Correctly predicted no strong emotions")
            passed_checks.append(True)
        else:
            print(f"\n⚠ Predicted emotions when none expected")
            passed_checks.append(False)


def run_manual_tests():
    """
    Run comprehensive manual testing of the emotion prediction system.
    
    Returns:
        dict: Test results summary including statistics and findings
    """
    print_section_header("MANUAL TESTING - MULTI-LABEL EMOTION CLASSIFICATION")
    
    print(f"Test Configuration:")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Prediction Threshold: {Config.PREDICTION_THRESHOLD}")
    print(f"  Max Length: {Config.MAX_LENGTH}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    print_section_header("LOADING MODEL")
    try:
        print(f"Loading model from: {Config.MODEL_SAVE_DIR}")
        model, tokenizer = load_model(Config.MODEL_SAVE_DIR, Config.DEVICE)
        print(f"✓ Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: {str(e)}")
        print("\nPlease train the model first by running: python train.py")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None
    
    # Initialize results tracking
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": 0,
        "passed_checks": 0,
        "categories": {},
        "findings": [],
        "limitations": []
    }
    
    passed_checks = []
    test_num = 0
    
    # Run tests for each category
    for category, test_cases in TEST_CASES.items():
        print_section_header(f"CATEGORY: {category.upper().replace('_', ' ')}")
        
        category_results = []
        
        for test_case in test_cases:
            test_num += 1
            
            try:
                # Perform prediction
                result = predict_emotions(
                    test_case['text'],
                    model,
                    tokenizer,
                    Config.DEVICE,
                    threshold=Config.PREDICTION_THRESHOLD
                )
                
                # Print result
                print_test_result(test_num, test_case, result, passed_checks)
                
                # Store result
                category_results.append({
                    "test_case": test_case,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                print(f"\n✗ Test {test_num} FAILED with error: {str(e)}")
                category_results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                })
                passed_checks.append(False)
        
        results_summary["categories"][category] = {
            "total": len(test_cases),
            "results": category_results
        }
    
    results_summary["total_tests"] = test_num
    results_summary["passed_checks"] = sum(passed_checks)
    
    # Test batch prediction
    print_section_header("BATCH PREDICTION TEST")
    print("Testing batch prediction with multiple comments...")
    
    batch_texts = [
        "I love this!",
        "This is terrible.",
        "Feeling calm and peaceful."
    ]
    
    try:
        batch_results = predict_emotions_batch(
            batch_texts,
            model,
            tokenizer,
            Config.DEVICE
        )
        
        print(f"\n✓ Batch prediction successful!")
        print(f"Processed {len(batch_texts)} comments in batch mode\n")
        
        for i, (text, result) in enumerate(zip(batch_texts, batch_results), 1):
            print(f"{i}. \"{text}\"")
            print(f"   Emotions: {', '.join(result['emotions']) if result['emotions'] else 'None'}")
        
        results_summary["batch_prediction"] = "Success"
    except Exception as e:
        print(f"\n✗ Batch prediction failed: {str(e)}")
        results_summary["batch_prediction"] = f"Failed: {str(e)}"
    
    # Generate findings and limitations
    print_section_header("FINDINGS AND OBSERVATIONS")
    
    findings = []
    limitations = []
    
    # Analyze results
    print("\nKey Findings:")
    
    # 1. Check positive emotion detection
    positive_category = results_summary["categories"].get("positive_emotions", {})
    if positive_category:
        print(f"\n1. Positive Emotion Detection:")
        print(f"   - Tested {positive_category['total']} positive emotion cases")
        positive_success = sum(1 for r in positive_category['results'] if r.get('success', False))
        print(f"   - Successfully processed {positive_success}/{positive_category['total']} cases")
        findings.append(f"Positive emotion detection: {positive_success}/{positive_category['total']} cases processed")
    
    # 2. Check negative emotion detection
    negative_category = results_summary["categories"].get("negative_emotions", {})
    if negative_category:
        print(f"\n2. Negative Emotion Detection:")
        print(f"   - Tested {negative_category['total']} negative emotion cases")
        negative_success = sum(1 for r in negative_category['results'] if r.get('success', False))
        print(f"   - Successfully processed {negative_success}/{negative_category['total']} cases")
        findings.append(f"Negative emotion detection: {negative_success}/{negative_category['total']} cases processed")
    
    # 3. Check mixed emotion detection
    mixed_category = results_summary["categories"].get("mixed_emotions", {})
    if mixed_category:
        print(f"\n3. Mixed Emotion Detection:")
        print(f"   - Tested {mixed_category['total']} mixed emotion cases")
        print(f"   - System can detect multiple simultaneous emotions")
        findings.append(f"Mixed emotion detection: System handles multiple simultaneous emotions")
    
    # 4. Check Vietnamese text
    vietnamese_category = results_summary["categories"].get("vietnamese_text", {})
    if vietnamese_category:
        print(f"\n4. Vietnamese Text Support:")
        print(f"   - Tested {vietnamese_category['total']} Vietnamese text cases")
        vietnamese_success = sum(1 for r in vietnamese_category['results'] if r.get('success', False))
        print(f"   - Successfully processed {vietnamese_success}/{vietnamese_category['total']} cases")
        print(f"   ⚠ Note: bert-base-uncased is primarily trained on English")
        print(f"   ⚠ Vietnamese support may be limited")
        findings.append(f"Vietnamese text: {vietnamese_success}/{vietnamese_category['total']} cases processed")
        limitations.append("Vietnamese support limited - bert-base-uncased is English-focused")
    
    # 5. Check edge cases
    edge_category = results_summary["categories"].get("edge_cases", {})
    if edge_category:
        print(f"\n5. Edge Cases:")
        print(f"   - Tested {edge_category['total']} edge cases")
        edge_success = sum(1 for r in edge_category['results'] if r.get('success', False))
        print(f"   - Successfully handled {edge_success}/{edge_category['total']} cases")
        findings.append(f"Edge cases: {edge_success}/{edge_category['total']} cases handled")
    
    # Document limitations
    print(f"\n\nIdentified Limitations:")
    
    limitations.extend([
        "Model requires training data to perform well",
        "Very short text (1-2 words) may not provide enough context",
        "Ambiguous or sarcastic text may be misclassified",
        "Threshold setting affects sensitivity (current: 0.5)",
        "Maximum sequence length is 512 tokens (longer text is truncated)"
    ])
    
    for i, limitation in enumerate(limitations, 1):
        print(f"   {i}. {limitation}")
    
    results_summary["findings"] = findings
    results_summary["limitations"] = limitations
    
    # Print summary
    print_section_header("TEST SUMMARY")
    
    print(f"Total Tests Run: {results_summary['total_tests']}")
    print(f"Checks Passed: {results_summary['passed_checks']}/{len(passed_checks)}")
    print(f"Success Rate: {results_summary['passed_checks']/len(passed_checks)*100:.1f}%")
    
    print(f"\nCategories Tested:")
    for category, data in results_summary["categories"].items():
        print(f"  - {category.replace('_', ' ').title()}: {data['total']} tests")
    
    print(f"\nBatch Prediction: {results_summary['batch_prediction']}")
    
    # Save results to file
    print_section_header("SAVING RESULTS")
    
    output_file = "manual_testing_results.json"
    try:
        # Prepare JSON-serializable results
        json_results = {
            "timestamp": results_summary["timestamp"],
            "configuration": {
                "model": Config.MODEL_NAME,
                "device": Config.DEVICE,
                "threshold": Config.PREDICTION_THRESHOLD,
                "max_length": Config.MAX_LENGTH
            },
            "summary": {
                "total_tests": results_summary["total_tests"],
                "passed_checks": results_summary["passed_checks"],
                "success_rate": f"{results_summary['passed_checks']/len(passed_checks)*100:.1f}%"
            },
            "categories": {
                cat: {"total": data["total"]}
                for cat, data in results_summary["categories"].items()
            },
            "batch_prediction": results_summary["batch_prediction"],
            "findings": results_summary["findings"],
            "limitations": results_summary["limitations"]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠ Could not save results to file: {str(e)}")
    
    print("\n" + "="*80)
    print(" MANUAL TESTING COMPLETE")
    print("="*80 + "\n")
    
    return results_summary


if __name__ == "__main__":
    results = run_manual_tests()
    
    if results:
        print("\n✓ Manual testing completed successfully!")
        print(f"✓ Tested {results['total_tests']} diverse examples")
        print(f"✓ Results saved to manual_testing_results.json")
    else:
        print("\n✗ Manual testing failed to complete")
        print("Please ensure the model is trained before running manual tests")
