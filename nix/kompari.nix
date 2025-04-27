{
  lib,
  fetchFromGitHub,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  pname = "kompari-cli";
  version = "0.1.0";
  src = fetchFromGitHub {
    owner = "linebender";
    repo = "kompari";
    rev = "4b851413e1b17307064aa48c50e59d7e29656543";
    hash = "sha256-kqDAITY3ndDpq7xvId0tmwgD0DEDbINF9vvtpsmb5qo=";
  };

  cargoHash = "sha256-R4EJskyzaHt+GLnRG1nioRmVr4EF7ILk9ufa+PojvgQ=";
}
