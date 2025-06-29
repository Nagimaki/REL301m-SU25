import numpy as np
import pygame

WIDTH = HEIGHT = 600
SIZE = (WIDTH, HEIGHT)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

ROWS = 3
COLS = 3
board = np.zeros((ROWS, COLS))


def draw_lines():
    # Draw horizontal and vertical lines to create the grid
    pygame.draw.line(window, BLACK, (0, HEIGHT // 3), (WIDTH, HEIGHT // 3), 5)
    pygame.draw.line(window, BLACK, (0, 2 * HEIGHT // 3), (WIDTH, 2 * HEIGHT // 3), 5)
    pygame.draw.line(window, BLACK, (WIDTH // 3, 0), (WIDTH // 3, HEIGHT), 5)
    pygame.draw.line(window, BLACK, (2 * WIDTH // 3, 0), (2 * WIDTH // 3, HEIGHT), 5)


def mark(row, col, player):
    if board[row, col] != 0:
        return False

    board[row, col] = player
    # Draw the player's mark on the board
    if player == 1:
        pygame.draw.circle(window, GREEN, (col * (WIDTH // 3) + WIDTH // 6, row * (HEIGHT // 3) + HEIGHT // 6), 90, 5)
    else:
        pygame.draw.line(
            window,
            RED,
            (col * (WIDTH // 3) + 20, row * (HEIGHT // 3) + 20),
            (col * (WIDTH // 3) + WIDTH // 3 - 20, row * (HEIGHT // 3) + HEIGHT // 3 - 20),
            5,
        )
        pygame.draw.line(
            window,
            RED,
            (col * (WIDTH // 3) + WIDTH // 3 - 20, row * (HEIGHT // 3) + 20),
            (col * (WIDTH // 3) + 20, row * (HEIGHT // 3) + HEIGHT // 3 - 20),
            5,
        )
    return True


def is_board_full():
    return np.all(board != 0)


gameover = False
turn = 0

pygame.init()
window = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Tic Tac Toe")
window.fill(WHITE)
draw_lines()
pygame.display.update()


def is_won_move(row, col, player):
    winning_color = GREEN if player == 1 else RED
    if np.all(board[row, :] == player):
        pygame.draw.line(
            window,
            winning_color,
            (0, row * (HEIGHT // 3) + HEIGHT // 6),
            (WIDTH, row * (HEIGHT // 3) + HEIGHT // 6),
            5,
        )
        return True
    if np.all(board[:, col] == player):
        pygame.draw.line(
            window,
            winning_color,
            (col * (WIDTH // 3) + WIDTH // 6, 0),
            (col * (WIDTH // 3) + WIDTH // 6, HEIGHT),
            5,
        )
        return True
    if np.all(np.diag(board) == player):
        pygame.draw.line(window, winning_color, (0, 0), (WIDTH, HEIGHT), 5)
        return True
    if np.all(np.diag(np.fliplr(board)) == player):
        pygame.draw.line(window, winning_color, (0, HEIGHT), (WIDTH, 0), 5)
        return True


while not gameover:
    player = 1 if turn % 2 == 0 else 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(event.pos)
            x, y = event.pos
            row = min(max(0, y // (HEIGHT // 3)), 2)
            col = min(max(0, x // (WIDTH // 3)), 2)
            print(row, col)

            if not mark(row, col, player):
                print("Cell already marked. Try again.")
                continue

            # Check for win
            if is_won_move(row, col, player):
                print(f"Player {player} wins!")
                gameover = True
            elif is_board_full():
                print("It's a draw!")
                gameover = True
            pygame.display.update()
            turn += 1

    if gameover:
        print("Game Over")
        pygame.time.delay(2000)
        board.fill(0)
        window.fill(WHITE)
        draw_lines()
        gameover = False
        pygame.display.update()
