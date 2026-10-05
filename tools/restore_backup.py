"""Reassemble and verify the archived volumes, or overlay a verified workspace delta."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import zipfile

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()

def contained(root, relative):
    name=PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe archive path: {relative}')
    root=root.resolve()
    target=root.joinpath(*name.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Path escapes target: {relative}')
    return target

def join(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    parts=[]
    for item in sorted(manifest['parts'],key=lambda x:x['order']):
        path=contained(args.parts_dir,item['name'])
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError(f'Volume checksum mismatch: {path.name}')
        parts.append(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as output:
        for path in parts:
            with path.open('rb') as source:
                shutil.copyfileobj(source,output,8*1024*1024)
    if args.output.stat().st_size!=manifest['bytes'] or digest(args.output)!=manifest['sha256']:
        raise ValueError('Reassembled archive checksum mismatch')
    print(f'Verified: {args.output}')

def overlay(args):
    with zipfile.ZipFile(args.delta) as archive:
        manifest=json.loads(archive.read('delta-manifest.json'))
        records=manifest['files']
        if isinstance(records,dict):
            records=[dict(value,path=key) for key,value in records.items()]
        # Validate every incoming member before replacing any workspace file.
        for item in records:
            relative=item.get('path',item.get('relative_path'))
            contained(args.root,relative)
            content=archive.read(relative)
            if len(content)!=item.get('bytes',item.get('size_bytes')) or hashlib.sha256(content).hexdigest()!=item['sha256']:
                raise ValueError(f'Delta checksum mismatch: {relative}')
        deletions=manifest.get('deleted',manifest.get('deleted_paths',[]))
        for relative in deletions:
            contained(args.root,relative)
        for item in records:
            relative=item.get('path',item.get('relative_path'))
            output=contained(args.root,relative)
            output.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(relative) as source,output.open('wb') as target:
                shutil.copyfileobj(source,target)
        if deletions and not args.apply_deletions:
            print(f'{len(deletions)} baseline files should be removed; rerun with --apply-deletions for exact restoration.')
        if args.apply_deletions:
            for relative in deletions:
                path=contained(args.root,relative)
                if path.is_file():path.unlink()
    print(f'Verified and applied {len(records)} delta files.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    command=commands.add_parser('join')
    command.add_argument('--manifest',type=Path,required=True)
    command.add_argument('--parts-dir',type=Path,required=True)
    command.add_argument('--output',type=Path,required=True)
    command.set_defaults(action=join)
    command=commands.add_parser('overlay')
    command.add_argument('--root',type=Path,required=True)
    command.add_argument('--delta',type=Path,required=True)
    command.add_argument('--apply-deletions',action='store_true')
    command.set_defaults(action=overlay)
    args=parser.parse_args()
    args.action(args)

if __name__=='__main__':main()
