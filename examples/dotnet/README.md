# RustODotnet consumer example

CI restores `RustODotnet` from current `build.yml` NuGet output, proving package
contents compile for a fresh consumer. For local work, pack first and restore from
that package directory:

```bash
dotnet pack ../../packages/dotnet/RustODotnet.csproj -c Release -o ./nupkg
dotnet restore RustODotnet.Example.sln --source ./nupkg --source https://api.nuget.org/v3/index.json
```

Then:

```bash
dotnet restore RustODotnet.Example.sln
dotnet build RustODotnet.Example.sln --configuration Release --no-restore
dotnet test RustODotnet.Example.sln --configuration Release --no-build
```

`RustODotnet.Example` demonstrates canonical `RustO.Initialize` and `DetectText`.
Without arguments it avoids native OCR. Running OCR needs packaged native runtime assets,
matching models, and an image path.

`RustODotnet.Example.Tests` only tests managed API contract. No model or native library needed.
