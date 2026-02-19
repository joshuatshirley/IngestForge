import pytest
from ingestforge.chunking.tree_sitter_chunker import TreeSitterCodeChunker


class TestTreeSitterCodeChunker:
    @pytest.fixture
    def chunker(self):
        return TreeSitterCodeChunker()

    def test_chunk_python(self, chunker):
        code = """
class MyClass:
    def __init__(self, x):
        self.x = x
    
    def my_method(self):
        return self.x

def my_function(y):
    return y + 1
"""
        records = chunker.chunk(code, "doc1", "test.py")

        # Expect: MyClass (as a whole if small), my_function
        # Or if MyClass is large, it would be MyClass header + methods
        # Here MyClass is small (under 4000 chars)

        kinds = [r.chunk_type for r in records]
        assert "code_class" in kinds
        assert "code_function" in kinds

        # Verify names
        titles = [r.section_title for r in records]
        assert "class MyClass" in titles
        assert "function my_function" in titles

    def test_chunk_javascript(self, chunker):
        code = """
class User {
    constructor(name) {
        this.name = name;
    }
}

function greet(user) {
    console.log("Hello " + user.name);
}

const add = (a, b) => a + b;
"""
        records = chunker.chunk(code, "doc2", "test.js")

        titles = [r.section_title for r in records]
        assert "class User" in titles
        assert "function greet" in titles
        assert "function add" in titles

    def test_chunk_unsupported_language(self, chunker):
        code = "Some random text in an unsupported file"
        records = chunker.chunk(code, "doc3", "test.txt")

        assert len(records) == 1
        assert records[0].chunk_type == "code_module"
