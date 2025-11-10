# LAMP - Latent Anti-Microbial Peptides

## Installation

Install with uv (workspace-aware):

```bash
# Base project only (default)
uv sync

# Install only the optional setup package
uv sync --package lamp-setup

# Install all workspace packages (root + setup)
uv sync --all-packages
```

## Secret Management with direnv

For managing environment variables and secrets (like `HF_TOKEN`), you can use [direnv](https://direnv.net/):

1. Install direnv:
   ```bash
   # macOS
   brew install direnv
   
   # Ubuntu/Debian
   apt install direnv
   ```

2. Hook direnv into your shell (add to your shell config):
   ```bash
   # For bash: add to ~/.bashrc
   eval "$(direnv hook bash)"
   
   # For zsh: add to ~/.zshrc
   eval "$(direnv hook zsh)"
   
   # For fish: add to ~/.config/fish/config.fish
   direnv hook fish | source
   ```

3. Create a `.envrc` file in the project root:
   ```bash
   export HF_TOKEN=hf_xxx
   ```

4. Allow direnv to load the file:
   ```bash
   direnv allow
   ```

Now your environment variables will be automatically loaded when you enter the project directory.
