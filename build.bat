@echo off
setlocal enabledelayedexpansion

if not exist build mkdir build

set SRCS=
for %%f in (src\*.cpp) do (
    set SRCS=!SRCS! %%f
)

g++ -std=c++17 -Wall -Wextra -O2 -g -static-libstdc++ -static-libgcc -o tlc.exe !SRCS!
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK
