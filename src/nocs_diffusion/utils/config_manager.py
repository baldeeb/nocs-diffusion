import pathlib as pl

class ConfigDirectoriesManager:
    def __init__(self, dir=None):
        if dir is None:
            pkg = pl.Path(__file__).parent.parent
            self.dir = pkg.resolve() / pl.Path('config')
        else:
            self.dir = dir
        self.files = [p for p in self.dir.glob('*.yaml')]

    def __getitem__(self, i):
        if type(i) == str:
            matches = [f for f in self.files if i == f.name]
            assert len(matches) > 0, "ERROR: no files of this name exist."
            assert len(matches) == 1, "ERROR: seems like there "+\
                        "are multiple configs of the same name"
            return matches[0]
        else:
            return str(self.files[i].absolute())
    
    def __str__(self):
        return '\n'.join([f.name for f in self.files])