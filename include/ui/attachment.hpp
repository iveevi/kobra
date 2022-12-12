#ifndef KOBRA_UI_ATTACHMENT_H_
#define KOBRA_UI_ATTACHMENT_H_

namespace kobra {

namespace ui {

// UI attachments for ImGUI rendering
struct ImGuiAttachment {
	virtual void render() = 0;
};

}

}

#endif
