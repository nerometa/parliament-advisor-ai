"""
Test script for Thai embedding model quality and token limits.

Tests:
1. Does model truncate silently on text > 32 tokens?
2. Quality check - cosine similarity between related/unrelated Thai articles
3. Fallback to multilingual-e5-large if Thai model fails

Model: KoonJamesZ/sentence-transformers-nina-thai-v3
Fallback: intfloat/multilingual-e5-large
"""

import sys
import warnings
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Model configurations
THAI_MODEL_NAME = "KoonJamesZ/sentence-transformers-nina-thai-v3"
FALLBACK_MODEL_NAME = "intfloat/multilingual-e5-large"

# Thai legal text samples for quality testing
# Similar texts - both about MP conflict of interest rules
SIMILAR_TEXTS = [
    "มาตรา ๑๕๗ สมาชิกสภาผู้แทนราษฎรจะต้องไม่เป็นสมาชิกวุฒิสภา และจะต้องไม่เป็นข้าราชการประจำ หรือเจ้าหน้าที่ของรัฐ",
    "มาตรา ๙๘ สมาชิกสภาผู้แทนราษฎรต้องไม่เป็นสมาชิกวุฒิสภา และห้ามมิให้ดำรงตำแหน่งในองค์กรของรัฐ",
]

# Truly dissimilar texts - different domains entirely
DISSIMILAR_TEXTS = [
    "มาตรา ๑๕๗ สมาชิกสภาผู้แทนราษฎรจะต้องไม่เป็นสมาชิกวุฒิสภา และจะต้องไม่เป็นข้าราชการประจำ หรือเจ้าหน้าที่ของรัฐ",
    "วันนี้อากาศดีมาก ไปเที่ยวทะเลกินข้าวผัดกระกะเพรา หมูกรอบอร่อยมาก",
]

# Token test samples (Thai text of varying lengths)
TOKEN_TEST_TEXTS = {
    10: "สมาชิกสภาผู้แทนต้องไม่เป็นข้าราชการ",
    32: "สมาชิกสภาผู้แทนราษฎรจะต้องไม่เป็นสมาชิกวุฒิสภา และจะต้องไม่เป็นข้าราชการประจำหรือเจ้าหน้าที่ของรัฐในเรื่องนี้",
    64: "สมาชิกสภาผู้แทนราษฎรจะต้องไม่เป็นสมาชิกวุฒิสภา และจะต้องไม่เป็นข้าราชการประจำหรือเจ้าหน้าที่ของรัฐ ในกรณีที่มีความขัดแย้งทางผลประโยชน์ สมาชิกสภาผู้แทนราษฎรต้องรายงานต่อประธานสภา และหลีกเลี่ยงการลงมติในเรื่องนั้น ๆ",
    128: "สมาชิกสภาผู้แทนราษฎรจะต้องไม่เป็นสมาชิกวุฒิสภา และจะต้องไม่เป็นข้าราชการประจำหรือเจ้าหน้าที่ของรัฐ ในกรณีที่มีความขัดแย้งทางผลประโยชน์ สมาชิกสภาผู้แทนราษฎรต้องรายงานต่อประธานสภา และหลีกเลี่ยงการลงมติในเรื่องนั้น ทั้งนี้เพื่อป้องกันการใช้อำนาจในทางที่ไม่เหมาะสม การฝ่าฝืนข้อกำหนดนี้อาจนำไปสู่การถูกลงโทษทางวินัยหรือการถูกถอดถอนจากตำแหน่ง โดยต้องมีการพิจารณาจากคณะกรรมการจริยธรรมของสภา และต้องได้รับความเห็นชอบจากสมาชิกส่วนใหญ่ การกระทำใดๆ ที่ก่อให้เกิดความเสียหายต่อเนื่องแก่รัฐหรือประชาชน ถือเป็นความผิดร้ายแรงที่ต้องได้รับการสอบสวนอย่างโปร่งใสและรวดเร็ว",
}


def load_model(model_name: str) -> Optional[SentenceTransformer]:
    """Load a sentence transformer model with error handling."""
    try:
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"  Successfully loaded: {model_name}")
        return model
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        return None


def test_token_limit(model: SentenceTransformer) -> dict:
    """Test how model handles texts of varying token lengths."""
    print("\n" + "=" * 60)
    print("TEST 1: Token Limit Behavior")
    print("=" * 60)
    
    results = {}
    
    for expected_tokens, text in TOKEN_TEST_TEXTS.items():
        print(f"\n  Testing with ~{expected_tokens} tokens:")
        print(f"    Text preview: {text[:50]}...")
        
        try:
            embedding = model.encode(text)
            embedding_dim = len(embedding)
            print(f"    SUCCESS: Embedding dimension = {embedding_dim}")
            results[expected_tokens] = {
                "status": "success",
                "embedding_dim": embedding_dim,
                "error": None
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            results[expected_tokens] = {
                "status": "error",
                "embedding_dim": None,
                "error": str(e)
            }
    
    # Analyze results
    print("\n  Token Limit Analysis:")
    has_success = any(r["status"] == "success" for r in results.values())
    has_error = any(r["status"] == "error" for r in results.values())
    
    if has_error and has_success:
        behavior = "PARTIAL - Some lengths work, some don't"
    elif has_error:
        behavior = "ERROR - Model fails on longer text"
    else:
        behavior = "OK - All token lengths work (model may truncate internally)"
    
    print(f"    Behavior: {behavior}")
    
    return {"results": results, "behavior": behavior}


def test_quality(model: SentenceTransformer, model_name: str) -> dict:
    """Test quality using cosine similarity between similar/dissimilar texts."""
    print("\n" + "=" * 60)
    print("TEST 2: Quality Check (Cosine Similarity)")
    print("=" * 60)
    
    results = {}
    
    # Test similar texts
    print("\n  Testing similar Thai legal texts:")
    for i, text in enumerate(SIMILAR_TEXTS):
        print(f"    Text {i+1}: {text[:40]}...")
    
    try:
        embeddings = model.encode(SIMILAR_TEXTS)
        sim_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"    Cosine similarity (similar): {sim_matrix:.4f}")
        similar_score = float(sim_matrix)
    except Exception as e:
        print(f"    ERROR: {e}")
        similar_score = None
    
    # Test dissimilar texts (truly different domains)
    print("\n  Testing dissimilar Thai texts (different domains):")
    for i, text in enumerate(DISSIMILAR_TEXTS):
        print(f"    Text {i+1}: {text[:40]}...")
    
    try:
        embeddings = model.encode(DISSIMILAR_TEXTS)
        dissim_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"    Cosine similarity (dissimilar): {dissim_matrix:.4f}")
        dissimilar_score = float(dissim_matrix)
    except Exception as e:
        print(f"    ERROR: {e}")
        dissimilar_score = None
    
    # Determine quality
    quality_pass = False
    recommendation = ""
    
    if similar_score is not None and dissimilar_score is not None:
        print(f"\n  Quality Analysis:")
        print(f"    Similar texts similarity: {similar_score:.4f} (target: > 0.7)")
        print(f"    Dissimilar texts similarity: {dissimilar_score:.4f} (target: < 0.3)")
        
        quality_pass = similar_score > 0.7 and dissimilar_score < 0.3
        print(f"    Quality test: {'PASS' if quality_pass else 'FAIL'}")
        
        if quality_pass:
            recommendation = f"USE {model_name} - Quality meets criteria"
        else:
            recommendation = f"REVIEW {model_name} - Quality below criteria"
    else:
        recommendation = f"ERROR with {model_name} - Could not compute quality"
    
    results = {
        "similar_score": similar_score,
        "dissimilar_score": dissimilar_score,
        "quality_pass": quality_pass,
        "recommendation": recommendation
    }
    
    print(f"\n  Recommendation: {recommendation}")
    
    return results


def main():
    """Main test function."""
    print("=" * 60)
    print("Thai Embedding Model Test Suite")
    print("=" * 60)
    
    final_recommendation = None
    thai_model = None
    fallback_model = None
    
    # Try to load Thai model
    print("\n[1] Loading primary model: KoonJamesZ/sentence-transformers-nina-thai-v3")
    thai_model = load_model(THAI_MODEL_NAME)
    
    if thai_model is None:
        print("\n  Thai model failed to load. Will use fallback.")
    else:
        # Test token limits
        token_results = test_token_limit(thai_model)
        
        # Test quality
        quality_results = test_quality(thai_model, THAI_MODEL_NAME)
        
        # Determine recommendation
        if quality_results.get("quality_pass"):
            final_recommendation = f"USE {THAI_MODEL_NAME}"
        else:
            final_recommendation = f"CONSIDER FALLBACK - Thai model quality below criteria"
    
    # Load fallback model for comparison/testing if Thai model had issues
    if thai_model is None or final_recommendation != f"USE {THAI_MODEL_NAME}":
        print("\n[2] Loading fallback model: intfloat/multilingual-e5-large")
        fallback_model = load_model(FALLBACK_MODEL_NAME)
        
        if fallback_model is not None:
            # Test token limits with fallback
            token_results_fb = test_token_limit(fallback_model)
            
            # Test quality with fallback
            quality_results_fb = test_quality(fallback_model, FALLBACK_MODEL_NAME)
            
            if quality_results_fb.get("quality_pass"):
                final_recommendation = f"USE FALLBACK - {FALLBACK_MODEL_NAME}"
            else:
                final_recommendation = "NO SUITABLE MODEL FOUND"
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Primary model: {THAI_MODEL_NAME}")
    print(f"  Fallback model: {FALLBACK_MODEL_NAME}")
    print(f"\n  RECOMMENDATION: {final_recommendation}")
    print("\n" + "=" * 60)
    
    return 0 if final_recommendation and "USE" in final_recommendation else 1


if __name__ == "__main__":
    sys.exit(main())
