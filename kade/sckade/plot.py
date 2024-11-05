import scanpy as sc
from matplotlib import rcParams


#---------------------------- Global configuration -----------------------------

def set_publication_params() -> None:
    """
    Set publication-level figure parameters
    """
    sc.set_figure_params(
        scanpy=True, dpi_save=600, vector_friendly=True, format="pdf",
        facecolor=(1.0, 1.0, 1.0, 0.0), transparent=False
    )
    rcParams["savefig.bbox"] = "tight"