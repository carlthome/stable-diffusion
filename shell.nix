with import <nixpkgs> {};
mkShell {
  packages = [
    direnv
    cudatoolkit
    cudaPackages.cudnn
    stdenv.cc.cc.lib
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${cudaPackages.cudnn}/lib:${cudatoolkit}/lib:$LD_LIBRARY_PATH
    direnv allow
  '';
}
