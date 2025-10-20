import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gudhi as gd

# ---- Instead of respective gudhi functions; to edit diagrams/barcodes more --- #

def plot_persistence_diagram(
    persistence=[],
    alpha=0.6,
    band=0.0,
    max_intervals=1000000,
    inf_delta=0.1,
    ax_data = [],
    legend=None,
    colormap=None,
    axes=None,
    fontsize=16,
    greyblock=True,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from gudhi.persistence_graphical_tools import _format_handler, _gudhi_matplotlib_use_tex, _limit_to_max_intervals, _min_birth_max_death, _matplotlib_can_use_tex


    if _gudhi_matplotlib_use_tex and _matplotlib_can_use_tex():
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
    else:
        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans")

    # By default, let's say the persistence is List[dimension, [birth, death]] - Can be from a persistence file
    input_type = 0
    

    try:
        persistence, input_type = _format_handler(persistence)
        persistence = _limit_to_max_intervals(
            persistence, max_intervals, key=lambda life_time: life_time[1][1] - life_time[1][0]
        )
        min_birth, max_death = _min_birth_max_death(persistence, band)
    except IndexError:
        min_birth, max_death = 0.0, 1.0
        pass

    delta = (max_death - min_birth) * inf_delta
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_end = max_death + delta / 2
    axis_start = min_birth - delta

    if ax_data != []:
        axis_start = ax_data[0]
        infinity = ax_data[1]
        axis_end = ax_data[2]
        delta = ax_data[3]
    

    if axes is None:
        _, axes = plt.subplots(1, 1)
    if colormap is None:
        colormap = plt.cm.Set1.colors
    # bootstrap band
    if band > 0.0:
        x = np.linspace(axis_start, infinity, 1000)
        axes.fill_between(x, x, x + band, alpha=alpha, facecolor="red")
    # lower diag patch
    if greyblock:
        axes.add_patch(
            mpatches.Polygon(
                [[axis_start, axis_start], [axis_end, axis_start], [axis_end, axis_end]],
                fill=True,
                color="lightgrey",
            )
        )
    # line display of equation : birth = death
    axes.plot([axis_start, axis_end], [axis_start, axis_end], linewidth=1.0, color="k")

    x = [birth for (dim, (birth, death)) in persistence]
    y = [death if death != float("inf") else infinity for (dim, (birth, death)) in persistence]
    c = [colormap[dim] for (dim, (birth, death)) in persistence]

    axes.scatter(x, y, alpha=alpha, color=c)
    if float("inf") in (death for (dim, (birth, death)) in persistence):
        # infinity line and text
        axes.plot([axis_start, axis_end], [infinity, infinity], linewidth=1.0, color="k", alpha=alpha)
        # Infinity label
        yt = axes.get_yticks()
        yt = yt[np.where(yt < axis_end)]  # to avoid plotting ticklabel higher than infinity
        yt = np.append(yt, infinity)
        ytl = ["%.3f" % e for e in yt]  # to avoid float precision error
        ytl[-1] = r"$+\infty$"
        axes.set_yticks(yt)
        axes.set_yticklabels(ytl)

    if legend is None and input_type != 1:
        # By default, if persistence is an array of (dimension, (birth, death)), or an
        # iterator[iterator[birth, death]], display the legend
        legend = True

    if legend:
        title = "Dimension"
        if input_type == 2:
            title = "Range"
        dimensions = list({item[0] for item in persistence})
        axes.legend(
            handles=[mpatches.Patch(color=colormap[dim], label=str(dim)) for dim in dimensions],
            title=title,
            loc="lower right",
        )

    axes.set_xlabel("Birth", fontsize=fontsize)
    axes.set_ylabel("Death", fontsize=fontsize)
    axes.set_title("Persistence diagram", fontsize=fontsize)
    # Ends plot on infinity value and starts a little bit before min_birth
    axes.axis([axis_start, axis_end, axis_start, infinity + delta / 2])
    return axes



def plot_persistence_barcode(
    persistence=[],
    alpha=0.6,
    max_intervals=20000,
    inf_delta=0.1,
    ax_data = [],
    legend=None,
    colormap=None,
    axes=None,
    fontsize=16,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from gudhi.persistence_graphical_tools import _format_handler, _gudhi_matplotlib_use_tex, _limit_to_max_intervals, _min_birth_max_death, _matplotlib_can_use_tex


    if _gudhi_matplotlib_use_tex and _matplotlib_can_use_tex():
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
    else:
        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans")

    # By default, let's say the persistence is List[dimension, [birth, death]] - Can be from a persistence file
    input_type = 0
    
    try:
        persistence, input_type = _format_handler(persistence)
        persistence = _limit_to_max_intervals(
            persistence, max_intervals, key=lambda life_time: life_time[1][1] - life_time[1][0]
        )
        (min_birth, max_death) = _min_birth_max_death(persistence)
        persistence = sorted(persistence, key=lambda birth: birth[1][0])
    except IndexError:
        min_birth, max_death = 0.0, 1.0
        pass

    delta = (max_death - min_birth) * inf_delta
    # Replace infinity values with max_death + delta for bar code to be more readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    if ax_data != []:
        axis_start = ax_data[0]
        infinity = ax_data[1]
        axis_end = ax_data[2]
        delta = ax_data[3]

    if axes is None:
        _, axes = plt.subplots(1, 1)
    if colormap is None:
        colormap = plt.cm.Set1.colors

    x = [birth for (dim, (birth, death)) in persistence]
    y = [(death - birth) if death != float("inf") else (infinity - birth) for (dim, (birth, death)) in persistence]
    c = [colormap[dim] for (dim, (birth, death)) in persistence]

    axes.barh(range(len(x)), y, height=3, left=x, alpha=alpha, color=c, linewidth=0)

    if legend is None and input_type != 1:
        # By default, if persistence is an array of (dimension, (birth, death)), or an
        # iterator[iterator[birth, death]], display the legend
        legend = True

    if legend:
        title = "Dimension"
        if input_type == 2:
            title = "Range"
        dimensions = {item[0] for item in persistence}
        axes.legend(
            handles=[mpatches.Patch(color=colormap[dim], label=str(dim)) for dim in dimensions],
            title=title,
            loc="best",
        )

    axes.set_title("Persistence barcode", fontsize=fontsize)
    axes.invert_yaxis()
    

    # Ends plot on infinity value and starts a little bit before min_birth
    if len(x) != 0:
        axes.set_xlim((axis_start, infinity))
        axes.axis([axis_start, infinity, axis_end, -3])
    return axes


# ---------------------------------------------- #





st.set_page_config(layout="wide")

plot_options = ['none', 'dino', 'away', 'h_lines', 'v_lines', 
                'x_shape', 'star', 'high_lines', 'dots', 
                'circle', 'bullseye', 'slant_up', 'slant_down', 
                'wide_lines']


if "language" not in st.session_state:
    st.session_state.language = "DE 🇩🇪"
if "rad" not in st.session_state:
    st.session_state.rad = 0
if "alpha" not in st.session_state:
    st.session_state.alpha = False
if "selectbox_0" not in st.session_state:
    st.session_state.selectbox_0 = "none"
if "selectbox_1" not in st.session_state:
    st.session_state.selectbox_1 = "none"





# --- LANGUAGE SWITCH HANDLER ---
def set_language(lang):
    st.session_state.language = lang


translations = {
    "EN 🇬🇧": {
        "title": "Topological data analysis as a tool for distinguishing datasets",
        "figs": "Figure ",
        "figs-header": "### Choose two data sets for analysis",
        "interpretation": "### Tools for interpretation",
        "alpha": "Show the alpha-complex",
        "goal": "### Goal of the analysis",
        "goal-text": "We wish to distinguish between data sets by quantifying their shape. Important signatures of shapes are their connected components (blue dots and bars 🔵) or holes (orange dots and bars 🟠).",
        "no-fig": "No figure selected",
        "stats": "**Simple statistics for** ",
        "source": "**Data source:** ",
        "author": "App created by Sophie Rosenmeier with ",
        "N": "Number of points: ",
        "mean-x": "Mean x-coord.: ",
        "mean-y": "Mean y-coord.: ",
        "std-x": "Std. x-coord.: ",
        "std-y": "Std. y-coord.: ",
        "pearson": "Pearson correlation: "

    },
    "DE 🇩🇪": {
        "title": "Topologische Datenanalyse als Unterscheidungshilfe für Datensätze",
        "figs": "Abbildung ",
        "figs-header": "### Wähle zwei Datensätze für die Analyse",
        "interpretation": "### Werkzeuge für Interpretation",
        "alpha": "Zeige den alpha-Komplex",
        "goal": "### Ziel der Analyse ",
        "goal-text": "Wir möchten Datensätze anhand ihrer Form unterscheiden. Wichtige Merkmale von Formen sind Zusammenhangskomponenten (blaue Punkte und Balken 🔵) oder Löcher (orange Punkte und Balken 🟠).",
        "no-fig": "Keine Abbildung ausgewählt",
        "stats": "**Einfache Statistik für** ",
        "source": "**Datenquelle:** ",
        "author": "App erstellt von Sophie Rosenmeier mit ",
        "N": "Anzahl Punkte: ",
        "mean-x": "Mittelwert x-Koord.: ",
        "mean-y": "Mittelwert y-Koord.: ",
        "std-x": "Standardabw. x-Koord.: ",
        "std-y": "Standardabw. y-Koord.: ",
        "pearson": "Pearson Korrelation: "
    }
}




# ------------ Different methods --------------- #

def plot_function(ax, plot_type):
    data = pd.read_csv('DatasaurusDozen.tsv', sep='\t')

    if plot_type == 'dino':
        x = data.groupby('dataset').get_group('dino')['x']
        y = data.groupby('dataset').get_group('dino')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('dino')
        return x, y

    elif plot_type == 'away':
        x = data.groupby('dataset').get_group('away')['x']
        y = data.groupby('dataset').get_group('away')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('away')
        return x, y

    elif plot_type == 'h_lines':
        x = data.groupby('dataset').get_group('h_lines')['x']
        y = data.groupby('dataset').get_group('h_lines')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('h_lines')
        return x, y

    elif plot_type == 'v_lines':
        x = data.groupby('dataset').get_group('v_lines')['x']
        y = data.groupby('dataset').get_group('v_lines')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('v_lines')
        return x, y

    elif plot_type == 'x_shape':
        x = data.groupby('dataset').get_group('x_shape')['x']
        y = data.groupby('dataset').get_group('x_shape')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('x_shape')
        return x, y

    elif plot_type == 'star':
        x = data.groupby('dataset').get_group('star')['x']
        y = data.groupby('dataset').get_group('star')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('star')
        return x, y

    elif plot_type == 'high_lines':
        x = data.groupby('dataset').get_group('high_lines')['x']
        y = data.groupby('dataset').get_group('high_lines')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('high_lines')
        return x, y

    elif plot_type == 'dots':
        x = data.groupby('dataset').get_group('dots')['x']
        y = data.groupby('dataset').get_group('dots')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('dots')
        return x, y

    elif plot_type == 'circle':
        x = data.groupby('dataset').get_group('circle')['x']
        y = data.groupby('dataset').get_group('circle')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('circle')
        return x, y

    elif plot_type == 'bullseye':
        x = data.groupby('dataset').get_group('bullseye')['x']
        y = data.groupby('dataset').get_group('bullseye')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('bullseye')
        return x, y

    elif plot_type == 'slant_up':
        x = data.groupby('dataset').get_group('slant_up')['x']
        y = data.groupby('dataset').get_group('slant_up')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('slant_up')
        return x, y

    elif plot_type == 'slant_down':
        x = data.groupby('dataset').get_group('slant_down')['x']
        y = data.groupby('dataset').get_group('slant_down')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('slant_down')
        return x, y
    
    elif plot_type == 'wide_lines':
        x = data.groupby('dataset').get_group('wide_lines')['x']
        y = data.groupby('dataset').get_group('wide_lines')['y']
        ax.plot(x, y, linestyle='none', marker='o', markersize=3, color='black')
        ax.set_title('wide_lines')
        return x, y

    return None, None


# Plotting persistence diagrams
def plot_pers_diag(ax, plot_type, x, y):
    pts = [*zip(x, y)] # fuse x- and y-coordinates together for gudhi
    cpx = gd.AlphaComplex(points=pts) # generating alpha complex filtration instead of cech complex filtration
    stree = cpx.create_simplex_tree(output_squared_values=False)
    pers_pairs = stree.persistence(homology_coeff_field=2)

    plot_persistence_diagram(pers_pairs, axes=ax, fontsize=10, colormap=([0.12,0.46,0.71], [1,0.5,0.05]), ax_data=[-0.5, 33, 34, 2], alpha=0.8)  
    plt.sca(ax)
    plt.xlabel('Birth')
    plt.ylabel('Death')

    ### For changing yticklabels from x.xxx to x
    labels = [item.get_text() for item in ax.get_yticklabels()]
    ax.set_yticklabels([str(round(float(label))) if label != '$+\\infty$' else '$+\\infty$' for label in labels])
    ###

    plt.title(plot_type)


# Plotting persistence barcodes
def plot_pers_bars(ax, plot_type, x, y):
    pts = [*zip(x, y)] 
    cpx = gd.AlphaComplex(points=pts) 
    stree = cpx.create_simplex_tree(output_squared_values=False)
    pers_pairs = stree.persistence(homology_coeff_field=2)

    plot_persistence_barcode(pers_pairs, axes=ax, colormap=([0.12,0.46,0.71], [1,0.5,0.05]), ax_data=[-0.5, 33, 265, 2], alpha=0.8)
    plt.sca(ax)
    plt.yticks([])
    plt.xlabel('Radius')
    plt.ylabel('Topological Features')
    plt.title(plot_type)


# Printing some common statistical parameters
def do_stats(plot_type, t):
    data = pd.read_csv('DatasaurusDozen.tsv', sep='\t')
    df = data.groupby('dataset').get_group(plot_type).drop('dataset', axis=1)

    st.write(t['N'], len(df))
    st.write(t['mean-x'], df.x.mean())
    st.write(t['std-x'], df.x.std())
    st.write(t['mean-y'], df.y.mean())
    st.write(t['std-y'], df.y.std())

    st.write(t['pearson'], df.corr(method='pearson').x.y)





def get_alpha_complex(ax, points, radius):
    alpha_complex = gd.AlphaComplex(points=points)
    stree = alpha_complex.create_simplex_tree(output_squared_values=False, max_alpha_square=radius**2)
    plt.sca(ax)
    
    # Plot edges and triangles
    for simplex, _ in stree.get_simplices():
        if len(simplex) == 2:
            i, j = simplex
            x = [points[i][0], points[j][0]]
            y = [points[i][1], points[j][1]]
            plt.plot(x, y, 'black')
        elif len(simplex) == 3:
            i, j, k = simplex
            triangle = plt.Polygon([points[i], points[j], points[k]],
                                edgecolor='black', fill=True, alpha=0.5)
            ax.add_patch(triangle)







# ----------- Main window ------------- #

main_window = st.container()

with main_window:

    _, col_lang = st.columns([0.85, 0.15])
    with col_lang:
        st.markdown("""
    <style>
    div[data-testid="stRadio"] > label {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

        languages = list(translations.keys())

        # --- Find current index in radio based on session_state.language ---
        current_index = next((i for i, lang in enumerate(languages) if lang == st.session_state.language), 0)
        # --- Radio button with no label and horizontal layout ---
        selected_label = st.radio(
        label="",
        options=languages,
        horizontal=True,
        index=current_index,
        key="language_radio"
    )

        # --- Update session state language based on selection ---
        st.session_state.language = next(
            lang for lang in languages if lang == selected_label
        )


        # --- ACTIVE TRANSLATION ---
        t = translations[st.session_state.language]

 
    st.title(t['title'])
    st.markdown(t['goal'])
    st.write(t['goal-text'])

    


    # Radius slider
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

    col_figs, _, col_interpretation = st.columns([0.5, 0.1, 0.4])

    with col_figs:
        st.markdown(t['figs-header'])
        current_pc_index = [plot_options.index(st.session_state.selectbox_0), plot_options.index(st.session_state.selectbox_1)]

        # Dropdowns (selectboxes) for selecting plots
        selected_plots = []
        for i in [0,1]:
            selection = st.selectbox(
                t['figs']+f"{i + 1}",
                plot_options,
                index=current_pc_index[i],
                key=f"selectbox_{i}"
            )
            selected_plots.append(selection)


    with col_interpretation:
        st.markdown(t['interpretation'])
        radius = st.slider(label="Radius", min_value=0, max_value=25, value=st.session_state.rad, step=1, key="rad")
        show_alpha = st.checkbox(t['alpha'], value=st.session_state.alpha, key="alpha")



    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

        


    row1_container = st.container()
    row2_container = st.container()

    for i, row in enumerate([row1_container, row2_container]):
        with row:
            if selected_plots[i] == 'none':
                    st.info(t['no-fig'])

            else:
                col1, col2 = st.columns([0.24, 0.76])

                with col1: 
                    #st.write("")
                    #st.markdown("<br>", unsafe_allow_html=True) 
                    #st.markdown("---")
                    st.markdown(t['stats']+f"**{selected_plots[i]}:**")
                    do_stats(selected_plots[i], t)
                    st.markdown("---")

                            

                with col2:
                    #st.markdown("<br>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
                    ax[0].set_ylim(bottom=-10, top=110)
                    ax[0].set_xlim(left=-10, right=110)
                    ax[0].axis('equal')
                    x, y = plot_function(ax[0], selected_plots[i])
                    x = list(x)
                    y = list(y)
                    points = list(zip(x,y))
                    for (x_val, y_val) in points:
                        circle = plt.Circle((x_val, y_val), radius, color='forestgreen', alpha=0.3)
                        ax[0].add_patch(circle)
                    if show_alpha == True:
                        get_alpha_complex(ax[0], points, radius)

                    plot_pers_diag(ax[1], selected_plots[i], x, y)
                    plt.sca(ax[1])
                    plt.plot([radius, radius], [radius, 36], color='forestgreen', linestyle='--')
                    plt.plot([0, radius], [radius, radius], color='forestgreen', linestyle='--')
                    plot_pers_bars(ax[2], selected_plots[i], x, y)
                    ax[2].axvline(x=radius, color='forestgreen', linestyle='--', label=f'x = {radius}')
                    st.pyplot(fig)


    impressum_container = st.container()

    with impressum_container:
        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
        st.markdown(t['source']+"[DataSaurus Dozen](https://www.research.autodesk.com/publications/same-stats-different-graphs/)"+", [Paper](https://www.research.autodesk.com/app/uploads/2023/03/same-stats-different-graphs.pdf_rec2hRjLLGgM7Cn2T.pdf)")
        st.markdown(t['author']+"[streamlit](https://streamlit.io), 2025.")









