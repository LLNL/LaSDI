#!/bin/bash

# Check whether Homebrew or wget is installed
if [ "$(uname)" == "Darwin" ]; then
  which -s brew > /dev/null
  if [[ $? != 0 ]] ; then
      # Install Homebrew
      echo "Homebrew installation is required."
      exit 1
  fi

  which -s wget > /dev/null
  # Install wget
  if [[ $? != 0 ]] ; then
      brew install wget
  fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$SCRIPT_DIR/dependencies
mkdir -p "$LIB_DIR"

# Install MFEM
cd "$LIB_DIR"
if [[ $BUILD_TYPE == "Debug" ]]; then
    if [ ! -d "mfem_debug" ]; then
    UPDATE_LIBS=true
    git clone https://github.com/mfem/mfem.git mfem_debug
    fi
    if [[ "$UPDATE_LIBS" == "true" ]]; then
        cd mfem_debug
        git pull
        make pdebug -j MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES METIS_DIR="$METIS_DIR" METIS_OPT="$METIS_OPT" METIS_LIB="$METIS_LIB"
    fi
    cd "$LIB_DIR"
    rm mfem
    ln -s mfem_debug mfem
else
    if [ ! -d "mfem_parallel" ]; then
      UPDATE_LIBS=true
      git clone https://github.com/mfem/mfem.git
    fi
    if [[ "$UPDATE_LIBS" == "true" ]]; then
    	cd mfem
    	git pull
   	make config
    	make serial -j
    fi
    cd "$LIB_DIR"
fi
