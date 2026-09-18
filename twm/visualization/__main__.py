"""One entry point for existing HDF5/release inspection and export."""


def main(argv=None):
    # Import playback/storage dependencies only when launching the viewer.
    from twm.visualize import main as playback_main
    return playback_main(argv)


if __name__ == "__main__":
    main()
