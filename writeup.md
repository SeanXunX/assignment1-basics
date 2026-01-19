# Problem (unicode1): Understanding Unicode (1 point)

## a

chr(0) returns `'\x00'`

## b

`"'\\x00'"`

## c

In python text, the `chr(0)` takes up one byte memory and exists within the string but it doesn't show up on screen when printed.

# Problem (unicode2): Unicode Encodings (3 points)

## a

UTF-8 try to use a more compactive way to encode characters. (e.g. It alwarys use one byte to encode ASCII characters.) While the other two ways waste too much space, as they add some redundant bytes.

## b

Example input: 你好

It fails because some characters are encoded into multiple bytes, and it cannot be decoded byte by byte.

## c

`b'\x80\x80'` utf-8 cannot decode byte start with 1 at the first bit.


