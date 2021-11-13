void _emit_line(vec4 start, vec4 end)
{
	gl_Position = start;
	EmitVertex();

	gl_Position = end;
	EmitVertex();

	EndPrimitive();
}