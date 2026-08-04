import argparse
import os

# MODELLER is a licensed academic tool; it builds the α/β linker loop. Import it lazily so
# `import kinapse` (and the whole structures package) works without it — only the linker step
# needs it. `LoopModel = object` lets the ScFvModel class body parse when MODELLER is absent;
# build_model() raises a clear, actionable error before anything MODELLER-specific runs.
_MODELLER_HINT = (
    "MODELLER is required to build the α/β linker (kinapse.structures.linkers) but is not "
    "installed / licensed.\n"
    "  1. install it into your env:  micromamba install -n kinapse -c salilab modeller\n"
    "     (or:  conda install -c salilab modeller)\n"
    "  2. get a free academic key:   https://salilab.org/modeller/registration.html\n"
    "  3. set the key (either):      export KEY_MODELLER=YOUR_KEY   (before first import)\n"
    "     or edit  <env>/lib/modeller-*/modlib/modeller/config.py  ->  license = 'YOUR_KEY'\n"
    "Then re-run. (The DiG linker step needs a real α/β-linked structure — see docs/README.)"
)
try:
    from modeller import Environ, selection
    from modeller.automodel import LoopModel, refine
    _MODELLER_OK = True
    _MODELLER_ERR = None
except Exception as _e:  # noqa: BLE001 - missing package OR missing licence
    _MODELLER_OK, _MODELLER_ERR = False, _e
    LoopModel = object   # placeholder so the class below still parses


class ScFvModel(LoopModel):
    def __init__(self, *args, linker_pos_start=None, linker_pos_end=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.linker_pos_start = int(linker_pos_start)
        self.linker_pos_end = int(linker_pos_end)

    def select_loop_atoms(self):
        # Only remodel the linker region
        return selection(self.residue_range(f'{self.linker_pos_start}:A', f'{self.linker_pos_end}:A'))


def build_model( linker_dir, alignment_file, model_name,
                linker_pos_start, linker_pos_end):
    if not _MODELLER_OK:
        raise ImportError(_MODELLER_HINT) from _MODELLER_ERR
    env = Environ()
    env.io.atom_files_directory = [linker_dir]
    a = ScFvModel(env,
                    alnfile=alignment_file,
                    knowns=model_name,
                    sequence=model_name + "_output",
                    linker_pos_start=linker_pos_start,
                    linker_pos_end=linker_pos_end)

    a.starting_model = 1
    a.ending_model = 1
    a.loop.starting_model = 1
    a.loop.ending_model = 1
    a.loop.md_level = refine.slow

    a.make()


if __name__=="__main__":
    #pass arguments
    MODELLER_KEY="MODELIRANJE"

    parser = argparse.ArgumentParser(description='Modeller for scFv')
    parser.add_argument('linker_dir', type=str, help='Directory containing linker')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('alignment_file', type=str, help='Alignment file')
    parser.add_argument('linked_pos_start', type=int, help='start of linker position')
    parser.add_argument('linked_pos_end', type=int, help='end of linker position')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    print(args.linked_pos_start, args.linked_pos_end)
    original_cwd = os.getcwd()
    os.chdir(args.output_dir)
    build_model( args.linker_dir, args.alignment_file, args.model_name, args.linked_pos_start, args.linked_pos_end)
    os.chdir(original_cwd)