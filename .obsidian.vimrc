" Have j and k navigate visual lines rather than logical ones
nnoremap j gj
nnoremap k gk
vnoremap j gj
vnoremap k gk

" I like using H and L for beginning/end of line
nnoremap H ^
nnoremap L $
vnoremap H ^
vnoremap L $

" Yank to system clipboard
set clipboard=unnamed

" Go back and forward with Ctrl+O and Ctrl+I
" (make sure to remove default Obsidian shortcuts for these to work)
exmap back obcommand app:go-back
nmap <C-i> :back<CR>
exmap forward obcommand app:go-forward
nmap <C-o> :forward<CR>

