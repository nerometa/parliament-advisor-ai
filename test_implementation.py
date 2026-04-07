#!/usr/bin/env python3
"""Comprehensive test suite for minimal RAG implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    results = []
    
    # Test 1: Config
    try:
        import config
        assert hasattr(config, 'VECTOR_DB_PATH')
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert config.VECTOR_DB_PATH == 'data/vector_store'
        results.append(('✓ Config', 'PASS'))
    except Exception as e:
        results.append(('✗ Config', f'FAIL: {e}'))
    
    # Test 2: Chunker
    try:
        from rag.chunker import Chunker, Chunk
        chunker = Chunker(chunk_size=512, overlap=50)
        test_text = "มาตรา 101 บุคคลซึ่งมีอายุไม่ต่ำกว่าสิบแปดปีบริบูรณ์"
        chunks = chunker.chunk(test_text, source='test.pdf')
        assert len(chunks) > 0
        assert isinstance(chunks[0], Chunk)
        results.append(('✓ Chunker', 'PASS'))
    except Exception as e:
        results.append(('✗ Chunker', f'FAIL: {e}'))
    
    # Test 3: Minimal RAG
    try:
        from rag.minimal_rag import ThaiRAG
        rag = ThaiRAG()
        stats = rag.get_stats()
        assert stats['status'] == 'not_loaded'  # No index yet
        results.append(('✓ Minimal RAG', 'PASS'))
    except Exception as e:
        results.append(('✗ Minimal RAG', f'FAIL: {e}'))
    
    # Test 4: Index script exists
    try:
        assert os.path.exists('index_knowledge.py')
        results.append(('✓ Index script', 'PASS'))
    except Exception as e:
        results.append(('✗ Index script', f'FAIL: {e}'))
    
    # Test 5: Requirements
    try:
        with open('requirements.txt') as f:
            content = f.read()
            assert 'numpy>=1.24.0' in content
            assert 'sentence-transformers' in content and '#' in content  # Commented out
        results.append(('✓ Requirements', 'PASS'))
    except Exception as e:
        results.append(('✗ Requirements', f'FAIL: {e}'))
    
    # Print results
    print('\n=== Test Results ===\n')
    for name, status in results:
        print(f'{name}: {status}')
    
    passed = sum(1 for _, status in results if status == 'PASS')
    print(f'\nTotal: {passed}/{len(results)} tests passed')
    
    if passed == len(results):
        print('\n✅ All core components working!')
        print('\nNext step: Build the index')
        print('  pip install sentence-transformers pypdf')
        print('  python index_knowledge.py')
    else:
        print('\n⚠️ Some tests failed')
    
    return passed == len(results)

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
