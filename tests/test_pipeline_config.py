import unittest

from rag_chatbot import get_pipeline_task


class PipelineTaskTests(unittest.TestCase):
    def test_t5_model_uses_text2text_generation(self):
        self.assertEqual(get_pipeline_task("google/flan-t5-base"), "text2text-generation")

    def test_causal_lm_model_uses_text_generation(self):
        self.assertEqual(get_pipeline_task("meta-llama/Llama-3.2-1B"), "text-generation")


if __name__ == "__main__":
    unittest.main()
