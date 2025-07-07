This document contains critical information about working with this codebase. Follow these guidelines precisely.

Depending on the task, refer to the research markdown files in the `research` folder and plan the task in detail before writing the code.

## Core Development Rules

- Run Python only in venv. Never run without activating the venv. Example: `source .venv/bin/activate && python main.py`
- ONLY use uv, NEVER pip
- Installation: uv add package
- Running tools: uv run tool
- Upgrading: uv add --dev package --upgrade-package package
- FORBIDDEN: uv pip install, @latest syntax

## Code Quality

- Type hints required for all code
- Public APIs must have docstrings
- Functions must be focused and small
- Follow existing patterns exactly
- Line length: 88 chars maximum

## Git

- Check git status before commits
- For commits fixing bugs or adding features, NEVER ever mention a co-authored-by or similar aspects. In particular, never mention the tool used to create the commit message or PR.

## CRITICAL NAMING RULES

**ABSOLUTELY FORBIDDEN SUFFIXES**: NEVER, EVER add these suffixes to file names or directories:
- `_optimized`, `_final`, `_enhanced`, `_improved`, `_better`, `_new`, `_updated`, `_fixed`, `_v2`, `_revised`, `_modified`, `_corrected`, `_refined`, `_polished`, `_clean`, `_working`, `_good`, `_best`
- ANY combination like `_final_fixed`, `_optimized_final`, `_enhanced_v2`, etc.
- camelCase variations like `Final`, `Optimized`, `Enhanced`, etc.

**RULE**: Use EXACTLY the file/directory names the user specifies. If they say `test_output`, use `test_output`. DO NOT "improve" or "enhance" the naming. The user chose the name for a reason.

**VIOLATION EXAMPLES** (DO NOT DO THIS):
- User says: `output` → Claude suggests: `output_optimized`
- User says: `test_data` → Claude suggests: `test_data_final`
- User says: `results` → Claude suggests: `results_enhanced` 

**CORRECT BEHAVIOR**:
- User says: `output` → Use: `output`
- User says: `test_data` → Use: `test_data`
- User says: `results` → Use: `results`

## Best Practices

- Keep changes minimal
- Follow existing patterns
- Work within the existing setup
- Always clean up test scripts like `debugging_*.py` or `test_*.py`
- Don't create new functions if you can edit the existing ones
- Don't create unnecessary files with suffixes like "_optimized", "_final", or camelCase names - work within existing files instead
- When presented with multiple options or methodologies, choose ONE appropriate approach - don't create complex scripts that try to implement all possibilities
- When proposing solution, don't jump to conclusions saying this will solve everything. You don't know until you run and see the output.
- NEVER truncate command output with pipes like `2>&1 | head -100` - always run commands with full output to get complete logs for debugging
- **HARDCODE FILE PATHS**: When user provides specific file names/paths in commands, hardcode them directly in the code instead of creating generic scripts with command line argument parsing. Work with the actual files mentioned, not placeholders.



## MacBook System Configuration

### Hardware Specifications

- **Model**: MacBook Pro (16-inch)
- **Model Identifier**: Mac16.5
- **Processor**: Apple M4 Max
  - Total CPU Cores: 16 (12 performance + 4 efficiency)
  - Total GPU Cores: 40
  - Total Neural Engine Cores: 16
- **Unified Memory**: 48 GB LPDDR5
- **Storage**: 1.0 TB SSD