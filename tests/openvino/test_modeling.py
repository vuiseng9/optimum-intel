#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import unittest
from unittest.mock import patch

import torch
from torchaudio.sox_effects import apply_effects_file
from datasets import load_dataset
from PIL import Image
import transformers
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.features import FeaturesManager, supported_features_mapping
from transformers.onnx import OnnxConfig
import importlib.util
from pathlib import Path

import requests
from evaluate import evaluator
from optimum.intel.openvino.modeling import (
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
)
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder, OVModelForSeq2SeqLM
from parameterized import parameterized


MODEL_NAMES = {
    "bart": "hf-internal-testing/tiny-random-bart",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "m2m_100": "valhalla/m2m100_tiny_random",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "t5": "hf-internal-testing/tiny-random-t5",
    "vit": "hf-internal-testing/tiny-random-vit",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    # "wav2vec2": "hf-internal-testing/tiny-random-wav2vec2",
    "wav2vec2": "anton-l/wav2vec2-base-ft-keyword-spotting"
}

SEED = 42


class OVModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True, compile=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        text = "This restaurant is awesome"
        outputs = pipe(text)
        self.assertTrue(model.is_dynamic)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)
        if model_arch == "bert":
            # Test FP16 conversion
            model.half()
            model.compile()
            outputs = pipe(text)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test static shapes
            model.reshape(1, 25)
            model.compile()
            outputs = pipe(text)
            self.assertTrue(not model.is_dynamic)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
        gc.collect()


class OVModelForQuestionAnsweringIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("start_logits" in ov_outputs)
        self.assertTrue("end_logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(ov_outputs.end_logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Arthur and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)
        gc.collect()

    def test_metric(self):
        model_id = "distilbert-base-cased-distilled-squad"
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data = load_dataset("squad", split="validation").select(range(50))
        task_evaluator = evaluator("question-answering")
        transformers_pipe = pipeline("question-answering", model=transformers_model, tokenizer=tokenizer)
        ov_pipe = pipeline("question-answering", model=ov_model, tokenizer=tokenizer)
        transformers_metric = task_evaluator.compute(model_or_pipeline=transformers_pipe, data=data, metric="squad")
        ov_metric = task_evaluator.compute(model_or_pipeline=ov_pipe, data=data, metric="squad")
        self.assertEqual(ov_metric["exact_match"], transformers_metric["exact_match"])
        self.assertEqual(ov_metric["f1"], transformers_metric["f1"])
        gc.collect()


class OVModelForTokenClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        gc.collect()


class OVModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("last_hidden_state" in ov_outputs)
        self.assertIsInstance(ov_outputs.last_hidden_state, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(
            torch.allclose(ov_outputs.last_hidden_state, transformers_outputs.last_hidden_state, atol=1e-4)
        )
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))
        gc.collect()


class OVModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("gpt2",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(f"This is a sample", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe(f"This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(["This is a sample" in item["generated_text"] for item in outputs]))
        gc.collect()


class OVModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForMaskedLM.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(f"This is a sample {tokenizer.mask_token}", return_tensors="pt")
        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForMaskedLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        outputs = pipe(f"This is a {tokenizer.mask_token}.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        gc.collect()


class OVModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("vit",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        ov_outputs = ov_model(**inputs)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        gc.collect()


class OVModelForSeq2SeqLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "marian",
        "mbart",
        "m2m_100",
        "t5",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)

        self.assertIsInstance(ov_model.encoder, OVEncoder)
        self.assertIsInstance(ov_model.decoder, OVDecoder)
        # self.assertIsInstance(ov_model.decoder_with_past, OVDecoder)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        ov_outputs = ov_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["translation_text"], str)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample input"
        tokens = tokenizer(text, return_tensors="pt")

        # General case
        outputs = model.generate(**tokens)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)

        # With input ids
        outputs = model.generate(input_ids=tokens["input_ids"])
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)

        gc.collect()

    def test_compare_with_and_without_past_key_values_model_outputs(self):
        model_id = MODEL_NAMES["t5"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample input"
        tokens = tokenizer(text, return_tensors="pt")
        model_with_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=True)
        outputs_model_with_pkv = model_with_pkv.generate(**tokens)
        model_without_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)
        outputs_model_without_pkv = model_without_pkv.generate(**tokens)
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))


# TODO(yujie): not finished
class OVModelForAudioClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("wav2vec2",)

    def apply_patch(self):  # TODO(yujie): use fixture
        spec = importlib.util.spec_from_file_location(
            "wav2vec2_onnx_config",
            Path(__file__).parents[2] / 'examples/openvino/audio-classification/wav2vec2_onnx_config.py'
        )
        wav2vec2_onnx_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wav2vec2_onnx_config)
        Wav2Vec2OnnxConfig = wav2vec2_onnx_config.Wav2Vec2OnnxConfig
        # Haven't figure out the way to patch. Using setattr temporarily.
        transformers.models.wav2vec2.Wav2Vec2OnnxConfig = Wav2Vec2OnnxConfig
        FeaturesManager._TASKS_TO_AUTOMODELS['audio-classification'] = AutoModelForAudioClassification
        FeaturesManager._SUPPORTED_MODEL_TYPE['wav2vec2'] = supported_features_mapping(
            "default",
            "audio-classification",
            onnx_config_cls='models.wav2vec2.Wav2Vec2OnnxConfig',
        )
        OnnxConfig._tasks_to_common_outputs["audio-classification"] = \
            {"input_values": {0: "batch", 1: "sequence"}}

    def get_dataset(self):  # TODO(yujie): use fixture
        effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

        def map_to_array(example):
            speech, _ = apply_effects_file(example["file"], effects)
            example["speech"] = speech.squeeze(0).numpy()
            print(example.shape)
            return example

        dataset = load_dataset("anton-l/superb_demo", "ks", split="test")
        dataset = dataset.map(map_to_array)
        return dataset

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.expectedFailure
    def test_compare_to_transformers(self, model_arch):
        self.apply_patch()
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        dataset = self.get_dataset()
        inputs = preprocessor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
        print(inputs['input_values'].shape)
        ov_outputs = ov_model(**inputs)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.expectedFailure
    def test_pipeline(self, model_arch):
        self.apply_patch()
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForAudioClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        # preprocessor.
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        dataset = self.get_dataset()
        outputs = pipe(dataset[:4]["speech"])
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        gc.collect()
