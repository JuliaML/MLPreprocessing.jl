using Documenter, MLPreprocessing

makedocs(
    modules = [MLPreprocessing],
    clean = false,
    format = :html,
    assets = [
        joinpath("assets", "favicon.ico"),
    ],
    sitename = "MLPreprocessing.jl",
    authors = "Christof Stocker, Andre Bieler",
    linkcheck = !("skiplinks" in ARGS),
    pages = Any[
        "Home" => "index.md",
        hide("Indices" => "indices.md"),
        "LICENSE.md",
    ],
    html_prettyurls = !("local" in ARGS),
)

deploydocs(
    repo = "github.com/JuliaML/MLPreprocessing.jl.git",
    target = "build",
    julia = "0.6",
    deps = nothing,
    make = nothing,
)
