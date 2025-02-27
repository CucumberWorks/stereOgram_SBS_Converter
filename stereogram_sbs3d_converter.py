# This file redirects imports to the core module
# It allows backward compatibility with existing code

from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

# Re-export the main class
__all__ = ['StereogramSBS3DConverter'] 