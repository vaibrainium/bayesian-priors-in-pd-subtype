class basic_utils:
    from .basic import load_model, raise_error, save_model


class plot_utils:
    from .plotting import (
        figure_by_height,
        figure_by_width,
        figure_with_cbar_by_height,
        grid_by_height,
        grid_by_width,
        plot_errorbar,
        plot_line,
        plot_scatter,
        plot_x_errorbar,
    )


class pmf_utils:
    from .pmf_utils import (
        fit_psychometric_function,
        get_accuracy_data,
        get_chronometric_data,
        get_psychometric_data,
    )


# Create a limited interface for glm_hmm_utils

__all__ = ["basic_utils", "plot_utils", "pmf_utils"]
