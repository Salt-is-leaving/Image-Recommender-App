#!/usr/bin/env python3
"""
Test FAISS Installation
Quick test to see if FAISS is properly installed
"""

def test_faiss():
    print("Testing FAISS installation...")
    
    try:
        import faiss
        print(f"✅ FAISS imported successfully!")
        print(f"   Version: {faiss.__version__}")
        print(f"   GPU support: {faiss.get_num_gpus() > 0}")
        print(f"   Available GPUs: {faiss.get_num_gpus()}")
        
        # Test basic functionality
        import numpy as np
        
        # Create test data
        d = 64
        nb = 100
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        
        # Create index
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        
        # Test search
        k = 4
        xq = np.random.random((1, d)).astype('float32')
        D, I = index.search(xq, k)
        
        print(f"✅ FAISS basic test passed!")
        print(f"   Index size: {index.ntotal}")
        print(f"   Search results shape: {I.shape}")
        return True
        
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_faiss()
    if success:
        print("\n🎉 FAISS is working correctly!")
    else:
        print("\n💡 FAISS issues detected. Try:")
        print("   conda uninstall faiss-cpu")
        print("   pip install faiss-cpu")
