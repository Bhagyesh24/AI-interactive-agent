import tkinter as tk

# Hardcoded test captions for the standalone demo
CAPTIONS = [
    ("What is blockchain?", "Blockchain is a secure, decentralized digital ledger that records transactions across many computers."),
]


def wrap_text(text, font, max_width, canvas):
    """Wrap text into lines that fit within max_width using the given font."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = current_line + " " + word if current_line else word
        bbox = canvas.bbox(canvas.create_text(0, 0, text=test, font=font, anchor="nw"))
        # bbox is (x1, y1, x2, y2); width = x2 - x1
        width = bbox[2] - bbox[0]
        canvas.delete("all")  # clear temporary text
        if width <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def main():
    root = tk.Tk()
    root.title("AI Companion Captions")
    root.geometry("800x150")
    root.configure(bg="black")
    root.attributes("-topmost", True)

    # Caption labels
    font = ("Arial", 18, "bold")
    text_color = "white"
    bg_color = "black"
    max_text_width = 760

    for question, answer in CAPTIONS:
        # Container for this caption set
        container = tk.Frame(root, bg=bg_color)
        container.pack(expand=True, fill="both", padx=20, pady=10)

        # You said line
        you_label = tk.Label(
            container,
            text=f"You said: {question}",
            font=font,
            fg=text_color,
            bg=bg_color,
            wraplength=max_text_width,
            justify="left",
            anchor="w",
        )
        you_label.pack(fill="x", pady=2)

        # AI line
        ai_label = tk.Label(
            container,
            text=f"AI: {answer}",
            font=font,
            fg=text_color,
            bg=bg_color,
            wraplength=max_text_width,
            justify="left",
            anchor="w",
        )
        ai_label.pack(fill="x", pady=2)

    # Auto-close after 10 seconds
    root.after(10000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
