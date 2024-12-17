class LinuxEmulator:
    def __init__(self):
        self.fs = {
            '/': {
                'type': 'dir',
                'children': {},
                'permissions': 'rwxr-xr-x'
            }
        }
        self.history = []
        self.env = {}
        self.is_root = False
        # Current directory always '/', for simplicity.
        self.cwd = '/'

    def _set_state(self, state):
        self.fs = state
        self.history = []
        self.env = {}
        self.is_root = False
        self.cwd = '/'

    def _log_history(self, cmd):
        self.history.append(cmd)

    def _split_path(self, path):
        if path.startswith('/'):
            parts = [p for p in path.split('/') if p]
        else:
            if self.cwd == '/':
                base_parts = []
            else:
                base_parts = [p for p in self.cwd.split('/') if p]
            rel_parts = [p for p in path.split('/') if p]
            parts = base_parts + rel_parts

        final_parts = []
        for p in parts:
            if p == '.':
                continue
            elif p == '..':
                if final_parts:
                    final_parts.pop()
            else:
                final_parts.append(p)
        return ['/' ] + final_parts if final_parts else ['/']

    def _get_node(self, path):
        parts = self._split_path(path)
        node = self.fs['/']
        for p in parts[1:]:
            if node['type'] != 'dir' or p not in node['children']:
                return None
            node = node['children'][p]
        return node

    def _ensure_dir(self, path):
        node = self._get_node(path)
        return node is not None and node['type'] == 'dir'

    def _make_path(self, path, node):
        if "~" in path:
            raise Exception("Unknown character '~'.")
        parts = self._split_path(path)
        parent_path_parts = parts[:-1]
        name = parts[-1]
        parent_node = self.fs['/']
        for p in parent_path_parts[1:]:
            if p not in parent_node['children']:
                parent_node['children'][p] = {
                    'type': 'dir',
                    'children': {},
                    'permissions': 'rwxr-xr-x'
                }
            parent_node = parent_node['children'][p]
            if parent_node['type'] != 'dir':
                raise Exception("Not a directory: " + '/'.join(parent_path_parts))
        parent_node['children'][name] = node

    def ls(self, path='/', all=False, show_perm=False):
        self._log_history("ls" + (" -a" if all else "") + (" -l" if show_perm else ""))
        dir_node = self._get_node(path)
        if not dir_node or dir_node['type'] != 'dir':
            return "ls: cannot access '{}': Not a directory".format(path)
        items = list(dir_node['children'].items())
        if not all:
            items = [(k,v) for k,v in items if not k.startswith('.')]
        items = sorted(items, key=lambda x: x[0])
        if show_perm:
            # Show permissions along with filenames
            return "\n".join(["{} {}".format(v['permissions'], k) for k,v in items])
        else:
            return "\n".join([k for k,v in items])

    def mkdir(self, path):
        self._log_history("mkdir " + path)
        node = self._get_node(path)
        if node is not None:
            raise Exception("mkdir: cannot create directory '{}': File exists".format(path))
        self._make_path(path, {
            'type': 'dir',
            'children': {},
            'permissions': 'rwxr-xr-x'
        })
        return ""

    def touch(self, path):
        self._log_history("touch " + path)
        node = self._get_node(path)
        if node is None:
            self._make_path(path, {
                'type': 'file',
                'content': "",
                'permissions': 'rw-r--r--'
            })
        else:
            if node['type'] == 'dir':
                raise Exception("touch: cannot touch '{}': Is a directory".format(path))
        return ""

    def grep(self, pattern, *files):
        cmd = "grep '{}' {}".format(pattern, ' '.join(files))
        self._log_history(cmd)
        if not files:
            return ""
        result_lines = []
        for f in files:
            node = self._get_node(f)
            if node is None or node['type'] != 'file':
                result_lines.append("grep: {}: No such file or directory".format(f))
                continue
            for line in node['content'].splitlines():
                if pattern in line:
                    prefix = (f + ":") if len(files) > 1 else ""
                    result_lines.append(prefix + line)
        return "\n".join(result_lines)

    def history_cmd(self):
        self._log_history("history")
        return "\n".join(self.history)

    def export_env(self, var, value):
        self._log_history("export {}={}".format(var, value))
        self.env[var] = value
        return ""

    def echo(self, *args):
        self._log_history("echo " + ' '.join(args))
        result_parts = []
        for arg in args:
            if arg.startswith('$'):
                var = arg[1:]
                result_parts.append(self.env.get(var, ""))
            else:
                result_parts.append(arg)
        return " ".join(result_parts)

    def cat(self, *files):
        self._log_history("cat " + ' '.join(files))
        result = []
        for f in files:
            node = self._get_node(f)
            if node is None:
                result.append("cat: {}: No such file or directory".format(f))
            elif node['type'] == 'dir':
                result.append("cat: {}: Is a directory".format(f))
            else:
                result.append(node['content'])
        return "\n".join(result)

    def write_overwrite(self, filename, content):
        self._log_history("> {} [content]".format(filename))
        node = self._get_node(filename)
        if node is None:
            self._make_path(filename, {
                'type': 'file',
                'content': content,
                'permissions': 'rw-r--r--'
            })
        else:
            if node['type'] == 'dir':
                raise Exception("bash: {}: Is a directory".format(filename))
            node['content'] = content
        return ""

    def write_append(self, filename, content):
        self._log_history(">> {} [content]".format(filename))
        node = self._get_node(filename)
        if node is None:
            self._make_path(filename, {
                'type': 'file',
                'content': content,
                'permissions': 'rw-r--r--'
            })
        else:
            if node['type'] == 'dir':
                raise Exception("bash: {}: Is a directory".format(filename))
            node['content'] += content
        return ""

    def rm(self, path, recursive=False, force=False):
        cmd = "rm"
        if recursive:
            cmd += " -r"
        if force:
            cmd += " -f"
        cmd += " " + path
        self._log_history(cmd)

        # Check if sudo is active
        if not self.is_root:
            raise Exception("rm: cannot remove '{}': Permission denied.".format(path))

        parts = self._split_path(path)
        name = parts[-1]
        parent_parts = parts[:-1]
        parent_node = self.fs['/']
        for p in parent_parts[1:]:
            if p not in parent_node['children'] or parent_node['children'][p]['type'] != 'dir':
                if force:
                    return ""
                raise Exception("rm: cannot remove '{}': No such file or directory".format(path))
            parent_node = parent_node['children'][p]
        if name not in parent_node['children']:
            if force:
                return ""
            raise Exception("rm: cannot remove '{}': No such file or directory".format(path))
        node = parent_node['children'][name]
        if node['type'] == 'dir':
            if not recursive:
                raise Exception("rm: cannot remove '{}': Is a directory".format(path))
        del parent_node['children'][name]
        return ""

    def sudo(self):
        self._log_history("sudo")
        self.is_root = True
        return ""

    def chmod(self, mode, path):
        self._log_history("chmod {} {}".format(mode, path))

        if type(mode) != str:
            raise TypeError(f"mode should be a permissions string but got {type(mode)}")
        assert len(mode) == 9, f"mode should be 9 characters long, but got {len(mode)}"
        # Check if sudo is active
        if not self.is_root:
            return "chmod: cannot change permissions of '{}': Permisson Denied.".format(path)

        node = self._get_node(path)
        if node is None:
            return "chmod: cannot access '{}': No such file or directory".format(path)
        node['permissions'] = mode
        return ""

    def mv(self, src, dest):
        self._log_history("mv {} {}".format(src, dest))
        src_node = self._get_node(src)
        if src_node is None:
            raise Exception("mv: cannot stat '{}': No such file or directory".format(src))

        # Determine the final path parts for src
        src_parts = self._split_path(src)
        src_name = src_parts[-1]
        src_parent_parts = src_parts[:-1]
        src_parent = self.fs['/']
        for p in src_parent_parts[1:]:
            src_parent = src_parent['children'][p]

        # Check if dest is a directory
        dest_node = self._get_node(dest)
        if dest_node is not None and dest_node['type'] == 'dir':
            # Move into directory
            dest_path_parts = self._split_path(dest)
            dest_parent = self.fs['/']
            for p in dest_path_parts[1:]:
                dest_parent = dest_parent['children'][p]

            # Move src_node into dest_node with the same name
            new_name = src_name
            # Remove from src parent
            del src_parent['children'][src_name]
            # Put into dest parent
            dest_parent['children'][new_name] = src_node
        else:
            # dest is not a directory, treat as rename
            dest_parts = self._split_path(dest)
            dest_name = dest_parts[-1]
            dest_parent_parts = dest_parts[:-1]

            # Find dest parent
            dest_parent = self.fs['/']
            for p in dest_parent_parts[1:]:
                if p not in dest_parent['children'] or dest_parent['children'][p]['type'] != 'dir':
                    raise Exception("mv: cannot move '{}' to '{}': No such directory".format(src, dest))
                dest_parent = dest_parent['children'][p]

            # Remove from src parent
            del src_parent['children'][src_name]
            # Put into dest parent with dest_name
            dest_parent['children'][dest_name] = src_node

        return ""
    
    def __eq__(self, value):

        try:
            return (self.fs == value.fs) and (self.env == value.env)
        except Exception as e:
            
            return False

_env = LinuxEmulator()

def look_in(path='/', all=False, show_perm=False):
    return _env.ls(path=path, all=all, show_perm=show_perm)

def lay_path(path, mode='file'):
    
    match mode:
        case 'file':
            _env.touch(path)
        case 'dir':
            _env.mkdir(path)
        case _:
            raise Exception(f"Invalid mode {mode}.")


def match_it(pattern, *files):
    return _env.grep(pattern, *files)

def look_back():
    return _env.history_cmd()

def export_env(var, value):
    return _env.export_env(var, value)

def callback(*args):
    return _env.echo(*args)

def view(*files):
    return _env.cat(*files)

def write_over(filename, content):
    return _env.write_overwrite(filename, content)

def write_next(filename, content):
    return _env.write_append(filename, content)

def destory(path, recursive=False, force=False):
    return _env.rm(path, recursive=recursive, force=force)

def super():
    return _env.sudo()

def mod_perms(mode, path):
    return _env.chmod(mode, path)

def move_it(src, dest):
    return _env.mv(src, dest)

def _set_state(state):
    _env._set_state(state)

QUESTIONS = [
    {
        "query": "Make my bashrc file in the cwd export MY_KEY as 123ASDGAS. Make sure to end with a newline.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    '.bashrc': {
                        'type': 'file',
                        'permissions': 'rwxr-xr-x',
                        'content': 'export DONT_OVERWRITE_ME=1',
                    }
                }
            },
            
        },
        "ground_truth_solution_steps": [
            "env.write_append('.bashrc', 'export MY_KEY=123ASDGAS\\n')"
        ]
    },
    {
        "query": "Make a .cache directory in the cwd. Inside the .cache directory make a cache.txt. Add 'hello world' to cache.txt.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {}
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('.cache')",
            "env.touch('.cache/cache.txt')",
            "env.write_overwrite('.cache/cache.txt', 'hello world')"
        ]
    },
    {
        "query": "In /, move all files that start with 'a' into a new directory, 'a_files'.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'apple.txt': {
                        'type': 'file',
                        'content': 'some apple content',
                        'permissions': 'rw-r--r--'
                    },
                    'archive.log': {
                        'type': 'file',
                        'content': 'error at line 1\nerror at line 2\n',
                        'permissions': 'rw-r--r--'
                    },
                    'banana.txt': {
                        'type': 'file',
                        'content': 'banana facts',
                        'permissions': 'rw-r--r--'
                    },
                    'docs': {
                        'type': 'dir',
                        'permissions': 'rwxr-xr-x',
                        'children': {
                            'readme.txt': {
                                'type': 'file',
                                'content': 'This is a readme file.',
                                'permissions': 'rw-r--r--'
                            }
                        }
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('a_files')",
            "env.mv('apple.txt', 'a_files')",
            "env.mv('archive.log', 'a_files')"
        ]
    },
    {
        "query": "In the CWD, move all files from the directory called 'current' to a new directory 'previous'. Delete the 'current' directory when you are done.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'current': {
                        'type': 'dir',
                        'permissions': 'rwxr-xr-x',
                        'children': {
                            'file1': {
                                'type': 'file',
                                'content': 'Data in file1',
                                'permissions': 'rw-r--r--'
                            },
                            'file2': {
                                'type': 'file',
                                'content': 'Data in file2',
                                'permissions': 'rw-r--r--'
                            }
                        }
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('previous')",
            "env.mv('current/file1', 'previous')",
            "env.mv('current/file2', 'previous')",
            "env.sudo()",
            "env.rm('current', recursive=True)"
        ]
    },
    {
        "query": "Give maximum permissions to the file 'shared.sys'. Backup the contents of shared.txt to a new file, shared_backup.txt.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'shared.sys': {
                        'type': 'file',
                        'content': 'System binary data...',
                        'permissions': 'rw-r--r--'
                    },
                    'shared.txt': {
                        'type': 'file',
                        'content': 'Shared information.\nLine2.\n',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.sudo()",
            "env.chmod('rwxrwxrwx', 'shared.sys')",
            "env.write_overwrite('shared_backup.txt', env.cat('shared.txt'))"
        ]
    },
    {
        "query": "Get the key from key.txt. Set the environment variable 'THE_KEY' to this value.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'key.txt': {
                        'type': 'file',
                        'content': '8asdxSDAxasd8123',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.cat('key.txt')",
            "env.export_env('THE_KEY', '8asdxSDAxasd8123')"
        ]
    },
    {
        "query": "Inside a new directory named 'alphabet_5' creating in the current working directory, add files for the first 5 letters of the alphabet: a.txt, b.txt, c.txt, d.txt, e.txt. Inside each file, should be the respective letter, and nothing else.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'children': {},
                'permissions': 'rwxr-xr-x'
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('alphabet_5')",
            "env.write_overwrite('alphabet_5/a.txt', 'a')",
            "env.write_overwrite('alphabet_5/b.txt', 'b')",
            "env.write_overwrite('alphabet_5/c.txt', 'c')",
            "env.write_overwrite('alphabet_5/d.txt', 'd')",
            "env.write_overwrite('alphabet_5/e.txt', 'e')"
        ]
    },
    # Additional 10 queries:
    {
        "query": "Create a directory named 'logs' in the CWD and inside it create two files: 'access.log' and 'error.log'. Append '404 not found' and add a newline to 'error.log'. Put the the contents of 'error.log' into an environment variable ERROR",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {}
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('logs')",
            "env.touch('logs/access.log')",
            "env.touch('logs/error.log')",
            "env.write_append('logs/error.log', '404 not found\\n')",
            "env.export_env('ERROR', env.cat('logs/error.log'))"
        ]
    },
    {
        "query": "Find all lines containing the word 'secret' in 'notes.txt' and set an environment variable called 'SECRET_COUNT' to the number of those lines.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'notes.txt': {
                        'type': 'file',
                        'content': 'This is a secret line.\nAnother line.\nsecret again here.\nNo secrets here.',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.export_env('SECRET_COUNT', 3)"
        ]
    },
    {
        "query": "Create a file called '.hiddenfile' in the CWD, and list all files including hidden ones along with their permissions. Store this output in an environment variable LS_OUTPUT.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'normal.txt': {
                        'type': 'file',
                        'content': 'Just some text',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.touch('.hiddenfile')",
            "env.export_env('LS_OUTPUT', env.ls(all=True, show_perm=True))"
        ]
    },
    {
        "query": "Change permissions of 'script.sh' to 'rwxr-xr-x'. Then run ls with permissions displayed. Store the output from ls in an environment variable LS_OUT.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'script.sh': {
                        'type': 'file',
                        'content': '#!/bin/bash\necho Hello World',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.sudo()",
            "env.chmod('rwxr-xr-x', 'script.sh')",
            "env.export_env('LS_OUT', env.ls(show_perm=True))"
        ]
    },
    {
        "query": "Create a directory 'backup' in the CWD. Move 'data.txt' into 'backup'. After it is moved, give data.txt the minimum permissions (no permissions).",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'data.txt': {
                        'type': 'file',
                        'content': 'Important data\n',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('backup')",
            "env.mv('data.txt', 'backup')",
            "env.sudo()",
            "env.chmod('---------','backup/data.txt')"
        ]
    },
    {
        "query": "Create a directory 'project' in the CWD with a subdirectory 'src'. Inside 'src', create files 'main.py' and 'utils.py' with no content. Then append 'import utils\\n' to 'main.py'.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {}
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('project')",
            "env.mkdir('project/src')",
            "env.touch('project/src/main.py')",
            "env.touch('project/src/utils.py')",
            "env.write_append('project/src/main.py', 'import utils\\n')"
        ]
    },
    {
        "query": "Set an environment variable 'DB_HOST' to the contents of network.sys.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'network.sys': {
                        'type': 'file',
                        'content': 'localhost',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.export_env('DB_HOST', env.cat('network.sys'))",
        ]
    },
    {
        "query": "Append a line 'export TOKEN=abc123' to '.bashrc'. Then make a copy of .bashrc called .bashrctwo.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    '.bashrc': {
                        'type': 'file',
                        'content': '# initial bashrc\n',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.write_append('.bashrc', 'export TOKEN=abc123\\n')",
            "env.export_env('TOKEN', 'abc123')",
            "env.write_append('.bashrctwo', env.cat('.bashrc'))"
        ]
    },
    {
        "query": "Create a directory 'archive' in the CWD. Move all '.log' files from the current directory into 'archive'. Then set and env var PERMS to the output of ls with permissions on the archive director.",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'app.log': {
                        'type': 'file',
                        'content': 'Application logs...',
                        'permissions': 'rw-r--r--'
                    },
                    'system.log': {
                        'type': 'file',
                        'content': 'System logs...',
                        'permissions': 'rw-r--r--'
                    },
                    'notes.txt': {
                        'type': 'file',
                        'content': 'Some notes',
                        'permissions': 'rw-r--r--'
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.mkdir('archive')",
            "env.mv('app.log', 'archive')",
            "env.mv('system.log', 'archive')",
            "env.export_env('PERMS', env.ls('/archive', show_perm=True))"
        ]
    },
    {
        "query": "Delete 'tempdir'",
        "initial_state": {
            '/': {
                'type': 'dir',
                'permissions': 'rwxr-xr-x',
                'children': {
                    'tempdir': {
                        'type': 'dir',
                        'permissions': 'rwxr-xr-x',
                        'children': {
                            'tempfile.txt': {
                                'type': 'file',
                                'content': 'Temporary data',
                                'permissions': 'rw-r--r--'
                            }
                        }
                    }
                }
            }
        },
        "ground_truth_solution_steps": [
            "env.sudo()",
            "env.rm('tempdir', recursive=True)"
        ]
    }
]

def _run_solution(initial_state, solution_steps):
    """
    Given an environment class (e.g. LinuxEmulator), an initial state (dict),
    and a list of solution step strings, this function:
    - Initializes the environment
    - Sets the state to initial_state
    - Executes each solution step line of code in order
    - Returns the final environment object
    """
    env = LinuxEmulator()
    env._set_state(initial_state)
    local_vars = {'env': env}
    for step in solution_steps:
        exec(step, {}, local_vars)
    return env

if __name__ == '__main__':
    from copy import deepcopy
    for task in QUESTIONS:
        assert _run_solution(initial_state=deepcopy(task["initial_state"]), solution_steps=task["ground_truth_solution_steps"]) == _run_solution(initial_state=deepcopy(task["initial_state"]), solution_steps=task["ground_truth_solution_steps"]), f"""Task: {task}\n\nState: {_run_solution(initial_state=task["initial_state"], solution_steps=task["ground_truth_solution_steps"]).fs}\n\nENV: {_run_solution(initial_state=task["initial_state"], solution_steps=task["ground_truth_solution_steps"]).env}"""

