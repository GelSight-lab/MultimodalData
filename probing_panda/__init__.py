try:
    from .displacement_data_collection import DispCollection
except ImportError:
    print("Error importing online modules in probing_panda module. Skipping.")