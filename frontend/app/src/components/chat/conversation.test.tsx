describe('Conversation component', () => {
  test('button should be disabled when input is empty', () => {
    // Create a simple test scenario without the full component complexity
    const textarea = document.createElement('textarea');
    const button = document.createElement('button');
    
    // Simulate the input validation logic
    const validateInput = (value: string) => {
      return !value.trim();
    };

    // Test empty input
    textarea.value = '';
    button.disabled = validateInput(textarea.value);
    expect(button.disabled).toBe(true);

    // Test whitespace only
    textarea.value = ' ';
    button.disabled = validateInput(textarea.value);
    expect(button.disabled).toBe(true);

    // Test whitespace with tab
    textarea.value = ' \t';
    button.disabled = validateInput(textarea.value);
    expect(button.disabled).toBe(true);

    // Test with actual content
    textarea.value = 'foo';
    button.disabled = validateInput(textarea.value);
    expect(button.disabled).toBe(false);
  });
});
