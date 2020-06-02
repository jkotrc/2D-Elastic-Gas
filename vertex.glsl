#version 400 core

in vec4 position;
uniform int WIDTH;
uniform int HEIGHT;


void main() {

	gl_Position = vec4(position.x/WIDTH, position.y/HEIGHT, 0.0f, 1.0f);

}