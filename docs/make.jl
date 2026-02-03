using Documenter
using DocumenterVitepress
using FunctionalGPs
using KernelFunctions

makedocs(;
    modules = [FunctionalGPs],
    authors = "Tim Weiland and contributors",
    sitename = "FunctionalGPs.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/timweiland/FunctionalGPs.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "timweiland.github.io/FunctionalGPs.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => [
            "Functionals" => "api/functionals.md",
            "Kernels" => "api/kernels.md",
            "Domains" => "api/domains.md",
            "Cross-Covariance" => "api/crosscov.md",
            "GP Conditioning" => "api/gps.md",
        ],
    ],
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/timweiland/FunctionalGPs.jl",
    push_preview = true,
)
