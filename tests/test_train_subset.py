import ast
import os
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from utils.train_subset import (make_manifest, validate_manifest, subset_label,
                               partial_output_dir, write_manifest, checkpoint_subset_label)

spec = importlib.util.spec_from_file_location('prepare_subset', Path(__file__).resolve().parents[1] / 'script/yuan/prepare_train_subset.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DummyImage:
    def __init__(self, index):
        self.index = index

    def convert(self, mode):
        return self

    def save(self, path):
        Path(path).write_text(str(self.index))


class Dataset:
    name = 'SLAKE'
    def __init__(self, size=101):
        self.samples = [{'id': i, 'question': f'q{i}'} for i in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return {'image': DummyImage(i), 'image_path': 'NA', 'query': f'q{i}', 'label': f'a{i}', 'prompt_template': 'Question: {}'}


class SubsetTests(unittest.TestCase):
    def test_deterministic_nested_and_limits(self):
        ds = Dataset()
        p = make_manifest(ds, 'SLAKE', 'vqa', fraction=.1)
        self.assertEqual(p, make_manifest(ds, 'SLAKE', 'vqa', fraction=.1))
        self.assertEqual(p['selected_size'], 10)
        self.assertEqual(len(set(p['indices'])), 10)
        self.assertTrue(set(p['indices']) <= set(make_manifest(ds, 'SLAKE', 'vqa', fraction=.2)['indices']))
        self.assertEqual(make_manifest(ds, 'SLAKE', 'vqa', maximum=200)['selected_size'], 101)
        self.assertEqual(make_manifest(ds, 'SLAKE', 'vqa', fraction=.0001)['selected_size'], 1)
        for kw in [{'fraction': 0}, {'fraction': float('nan')}, {'maximum': 0}, {'maximum': -1}, {'fraction': .1, 'maximum': 5}]:
            with self.assertRaises(ValueError): make_manifest(ds, 'SLAKE', 'vqa', **kw)
        with self.assertRaises(ValueError): make_manifest(Dataset(0), 'SLAKE', 'vqa')

    def test_manifest_detects_wrong_source_and_indices(self):
        ds = Dataset()
        p = make_manifest(ds, 'SLAKE', 'vqa', maximum=5)
        self.assertEqual(validate_manifest(p, ds, 'SLAKE', 'vqa'), p['indices'])
        for key, value in [('dataset', 'PathVQA'), ('split', 'test'), ('task', 'diagnosis'), ('selected_size', 9), ('indices', [0]*5), ('indices', [-1]*5)]:
            with self.assertRaises(ValueError): validate_manifest(dict(p, **{key: value}), ds, 'SLAKE', 'vqa')
        ds.samples.reverse()
        with self.assertRaises(ValueError): validate_manifest(p, ds, 'SLAKE', 'vqa')

    def test_paths_and_manifest_collision(self):
        self.assertEqual(partial_output_dir('/exp', 'vqa', 'SLAKE', 'Quilt-LLaVA', 'train_lora', subset_label(.1)), '/exp/vqa/SLAKE/partial-data-exp/10pct/Quilt-LLaVA/train_lora')
        self.assertEqual(partial_output_dir('/exp', 'vqa', 'SLAKE', 'M', 'train', subset_label()), '/exp/vqa/SLAKE/M/train')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'manifest.json'
            p = make_manifest(Dataset(), 'SLAKE', 'vqa', maximum=5)
            write_manifest(path,p);write_manifest(path,p)
            self.assertEqual(subset_label(manifest=path), '5samples')
            with self.assertRaises(ValueError): write_manifest(path, dict(p, fraction_seed=3))
            with self.assertRaises(ValueError): subset_label(fraction=.1, manifest=path)

    def test_native_wrapper_uses_saved_indices_each_epoch(self):
        # Exercise the real factory function without importing GPU/model dependencies.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'dataset/__init__.py').read_text())
        definitions = [node for node in tree.body if getattr(node, 'name', None) in {'FractionalDataset', '_apply_train_fraction'}]
        fake_torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)))
        ns = {'torch': fake_torch, 'os': os, 'json': json}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<native-subset>', 'exec'), ns)
        ds = Dataset()
        with tempfile.TemporaryDirectory() as td:
            manifest = make_manifest(ds, 'SLAKE', 'vqa', maximum=9)
            path = Path(td) / 'shared.json'
            write_manifest(path, manifest)
            args = SimpleNamespace(dataset='SLAKE', task='vqa', train_subset_manifest=str(path), output_dir=td)
            subset = ns['_apply_train_fraction'](ds,args,'train')
            for _ in range(3):
                self.assertEqual([subset[i]['query'] for i in range(len(subset))], [f'q{i}' for i in manifest['indices']])
            self.assertIs(ns['_apply_train_fraction'](ds,SimpleNamespace(),'test'), ds)
            with self.assertRaises(ValueError): ns['_apply_train_fraction'](ds,args,'test')

    def test_partial_eval_paths_and_seeds(self):
        self.assertEqual(subset_label(.1,seed=43), '10pct-fseed43')
        path = '/exp/vqa/SLAKE/partial-data-exp/10pct/M/train/checkpoint-500'
        self.assertEqual(checkpoint_subset_label(path),'10pct')
        self.assertIsNone(checkpoint_subset_label('/pretrained/base'))
        self.assertEqual(partial_output_dir('/exp','vqa','SLAKE','M','eval_seed42',checkpoint_subset_label(path)),
                         '/exp/vqa/SLAKE/partial-data-exp/10pct/M/eval_seed42')

    def test_cross_framework_records(self):
        ds = Dataset(10)
        p = make_manifest(ds,'SLAKE','vqa',maximum=7)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root/'temp_images_train'
            images.mkdir()
            records = []
            for i in range(len(ds)):
                image = images/f'{i}_0.jpg'
                image.write_bytes(b'existing JPEG placeholder')
                records.append({'conversations': [{'from':'human','value':f'<image>Question: q{i}'},
                                                   {'from':'gpt','value':f'a{i}'}], 'images':[str(image)]})
            source = root/'original.json'
            source.write_text(json.dumps(records))
            before = source.read_bytes()
            prepare.export_sharegpt(ds,p,root/'subset',source)
            actual = json.loads((root/'subset/train.json').read_text())
            self.assertEqual(actual, [records[i] for i in p['indices']])
            self.assertFalse((root/'subset/images').exists())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(list(images.iterdir())),10)
            self.assertTrue(all(f.read_bytes()==b'existing JPEG placeholder' for f in images.iterdir()))
            info=json.loads((root/'subset/dataset_info.json').read_text())
            self.assertEqual(info['shared_subset']['columns']['messages'],'conversations')
            # Wrong count, changed question, wrong image reference and missing image
            # must fail without creating the destination or silently remapping rows.
            index = p['indices'][0]
            for case in ['count','question','image','missing']:
                bad=json.loads(before)
                if case=='count': bad.pop()
                elif case=='question': bad[index]['conversations'][0]['value']='changed'
                elif case=='image': bad[index]['images']=[records[(index+1)%10]['images'][0]]
                else: bad[index]['images']=[str(root/'missing.jpg')]
                source.write_text(json.dumps(bad))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    prepare.export_sharegpt(ds,p,root/case,source)
                self.assertFalse((root/case).exists())

    def test_legacy_single_image_names(self):
        ds = Dataset(2)
        payload = make_manifest(ds, 'SLAKE', 'vqa', maximum=2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / 'temp_images_train'
            images.mkdir()
            records = []
            for i in range(2):
                image = images / f'{i}.jpg'
                image.write_bytes(b'original image')
                records.append({'conversations': [{'from': 'human', 'value': f'<image>Question: q{i}'},
                                                   {'from': 'gpt', 'value': f'a{i}'}], 'images': [str(image)]})
            source = root / 'train.json'
            source.write_text(json.dumps(records))
            prepare.export_sharegpt(ds, payload, root / 'subset', source)
            self.assertEqual(json.loads((root / 'subset/train.json').read_text()), records)
            records[0]['images'] = records[1]['images']
            source.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'image reference mismatch'):
                prepare.export_sharegpt(ds, payload, root / 'wrong-index', source)

if __name__ == '__main__': unittest.main()
