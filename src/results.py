"""Shared HDF5 metadata reading for the Python and Julia interfaces."""


def text_attribute(group, name):
    """
    Read text written as a Python string or a Julia HDF5 byte string.

    # Arguments:
    - group::h5py.Group: The file or group.
    - name::str: The attribute name.

    # Returns:
    - str or None: The decoded value, if present.
    """
    value = group.attrs.get(name)
    return value.decode('utf-8') if isinstance(value, bytes) else value
